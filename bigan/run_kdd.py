import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import bigan.kdd_utilities as network
import data.kdd as data
from sklearn.metrics import precision_recall_fscore_support

RANDOM_SEED = 13
FREQ_PRINT = 20  # print frequency image tensorboard [20]


def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter


def display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree):
    '''See parameters
    '''
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Method for discriminator: ', method)
    print('Degree for L norms: ', degree)


def display_progression_epoch(j, id_max):
    '''See epoch progression
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def create_logdir(method, weight, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "bigan/train_logs/kdd/{}/{}/{}".format(weight, method, rd)


# example: 模型，bigan或者gan  dataset:数据集，kdd或minist   split: 固定为run  nb_epochs: 迭代轮次数 label: 异常标签 w: 映射损失权重 m: 鉴别器损失函数类型 degree:特征匹配损失   rd:随机数种子
def train_and_test(nb_epochs, weight, method, degree, random_seed):
    """ Runs the Bigan on the KDD dataset

    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        nb_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        method (str, optional): 'fm' for ``Feature Matching`` or "cross-e"
                                     for ``cross entropy``, "efm" etc.
        anomalous_label (int): int in range 0 to 10, is the class/digit
                                which is considered outlier
    """
    logger = logging.getLogger("BiGAN.train.kdd.{}".format(method))

    # Placeholders 是一个占位符,通俗的来说就是先定义一个变量形参,然后再sess会话里面赋值进去! 后面通过feed_dict防止数据进来。。
    input_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input")  # (None, 121)
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    trainx, trainy = data.get_train()
    trainx_copy = trainx.copy()
    testx, testy = data.get_test()

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.9999  # 用来提升效果的？

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    logger.info('Building training graph...')

    logger.warn("The BiGAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    gen = network.decoder
    enc = network.encoder
    dis = network.discriminator
    # 在某个tf.name_scope()指定的区域中定义的所有对象及各种操作，他们的name属性上会增加该命名域的区域名，用于区别对象属于哪个区域
    with tf.variable_scope('encoder_model'):
        z_gen = enc(input_pl, is_training=is_training_pl)

    with tf.variable_scope('generator_model'):
        z = tf.random_normal(
            [batch_size, latent_dim])  # tf.random_normal()函数用于从“服从指定正态分布的序列”中随机取出指定个数的值,即生成batch_size行 latent_dim列的随机数。
        x_gen = gen(z, is_training=is_training_pl)

    with tf.variable_scope('discriminator_model'):
        l_encoder, inter_layer_inp = dis(z_gen, input_pl, is_training=is_training_pl)
        l_generator, inter_layer_rct = dis(z, x_gen, is_training=is_training_pl, reuse=True)

    with tf.name_scope('loss_functions'):
        # discriminator,编码器的数据判别为真，生成器的数据判别为假，取二者之和
        loss_dis_enc = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder), logits=l_encoder))
        loss_dis_gen = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator), logits=l_generator))
        loss_discriminator = loss_dis_gen + loss_dis_enc
        # generator，生成器损失函数，目标是让鉴别器判别为真
        loss_generator = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator), logits=l_generator))
        # encoder， 编码器损失函数，让鉴别器判断假
        loss_encoder = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder), logits=l_encoder))

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]
        # 定义优化器
        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')
        # 更新参数
        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer_enc.minimize(loss_encoder, var_list=evars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(loss_discriminator, var_list=dvars)

        # Exponential Moving Average for estimation
        """
        移动平均就是用当前时刻前期的观测值预测下一期的取值。 Xt可预测前n期观测值，即前n期取平均。
        指数移动平为以前的数值分配了一系列固定的指数递减权重，即权重系数随着时间呈指数下降。EMA提供了一个更明显的指标，能更快地反映最近的价格趋势。即(a1x1+a2x3+..+anxn) / (1+ 1-a1 + (1-a2)2 +..+ (1-an)n ),即越靠近
        观测值权重越大。ema_decay即衰减率，表征取前多少个数值作为预测基础，越大越趋于稳定，0.99大约是前100个
        """
        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        # 采用滑动平均的方法更新参数
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

        enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_enc = enc_ema.apply(evars)

        with tf.control_dependencies([enc_op]):
            train_enc_op = tf.group(maintain_averages_op_enc)
    # tf.summary()能够保存训练过程以及参数分布图并在tensorboard显示
    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_discriminator', loss_discriminator, ['dis'])
            tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])  # 用于显示标量信息
            tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_generator, ['gen'])
            tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])

        sum_op_dis = tf.summary.merge_all('dis')  # 可以将所有已定义的summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可显示训练时的各种信息了
        sum_op_gen = tf.summary.merge_all('gen')

    logger.info('Building testing graph...')

    with tf.variable_scope('encoder_model'):
        z_gen_ema = enc(input_pl, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True)

    with tf.variable_scope('generator_model'):
        reconstruct_ema = gen(z_gen_ema, is_training=is_training_pl,
                              getter=get_getter(gen_ema), reuse=True)

    with tf.variable_scope('discriminator_model'):
        l_encoder_ema, inter_layer_inp_ema = dis(z_gen_ema,
                                                 input_pl,
                                                 is_training=is_training_pl,
                                                 getter=get_getter(dis_ema),
                                                 reuse=True)
        l_generator_ema, inter_layer_rct_ema = dis(z_gen_ema,
                                                   reconstruct_ema,
                                                   is_training=is_training_pl,
                                                   getter=get_getter(dis_ema),
                                                   reuse=True)
    # 这一步计算异常得分，论文的方法是 (1 - weight) * gen_score + weight * dis_score
    with tf.name_scope('Testing'):
        # 这里计算得到残差
        with tf.variable_scope('Reconstruction_loss'):
            delta = input_pl - reconstruct_ema
            delta_flat = tf.contrib.layers.flatten(delta)
            gen_score = tf.norm(delta_flat, ord=degree, axis=1,
                                keep_dims=False, name='epsilon')
        # 这里是鉴别器的鉴别得分
        with tf.variable_scope('Discriminator_loss'):
            if method == "cross-e":
                dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(l_generator_ema), logits=l_generator_ema)

            elif method == "fm":
                fm = inter_layer_inp_ema - inter_layer_rct_ema
                fm = tf.contrib.layers.flatten(fm)
                dis_score = tf.norm(fm, ord=degree, axis=1,
                                    keep_dims=False, name='d_loss')
            # 删除维度是1的那个维度
            dis_score = tf.squeeze(dis_score)

        with tf.variable_scope('Score'):
            list_scores = (1 - weight) * gen_score + weight * dis_score

    logdir = create_logdir(weight, method, random_seed)

    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None,
                             save_model_secs=120)

    logger.info('Start training...')
    with sv.managed_session() as sess:

        logger.info('Initialization done')
        writer = tf.summary.FileWriter(logdir, sess.graph)
        train_batch = 0
        epoch = 0

        while not sv.should_stop() and epoch < nb_epochs:

            lr = starting_lr
            begin = time.time()

            # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]
            train_loss_dis, train_loss_gen, train_loss_enc = [0, 0, 0]

            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                feed_dict = {input_pl: trainx[ran_from:ran_to],
                             is_training_pl: True,
                             learning_rate: lr}

                _, ld, sm = sess.run([train_dis_op,
                                      loss_discriminator,
                                      sum_op_dis],
                                     feed_dict=feed_dict)
                train_loss_dis += ld
                writer.add_summary(sm, train_batch)

                # train generator and encoder
                feed_dict = {input_pl: trainx_copy[ran_from:ran_to],
                             is_training_pl: True,
                             learning_rate: lr}
                _, _, le, lg, sm = sess.run([train_gen_op,
                                             train_enc_op,
                                             loss_encoder,
                                             loss_generator,
                                             sum_op_gen],
                                            feed_dict=feed_dict)
                train_loss_gen += lg
                train_loss_enc += le
                writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_enc /= nr_batches_train
            train_loss_dis /= nr_batches_train

            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_enc, train_loss_dis))

            epoch += 1

        logger.warn('Testing evaluation...')

        inds = rng.permutation(testx.shape[0])
        testx = testx[inds]  # shuffling  dataset
        testy = testy[inds]  # shuffling  dataset
        scores = []
        inference_time = []

        # Create scores
        for t in range(nr_batches_test):
            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_val_batch = time.time()

            feed_dict = {input_pl: testx[ran_from:ran_to],
                         is_training_pl: False}
            # 得到每个批次的得分数组
            scores += sess.run(list_scores,
                               feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_val_batch)

        logger.info('Testing : mean inference time is %.4f' % (
            np.mean(inference_time)))  # 计算平均值

        ran_from = nr_batches_test * batch_size
        ran_to = (nr_batches_test + 1) * batch_size
        size = testx[ran_from:ran_to].shape[0]
        fill = np.ones([batch_size - size, 121])

        batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
        feed_dict = {input_pl: batch,
                     is_training_pl: False}

        batch_score = sess.run(list_scores,
                               feed_dict=feed_dict).tolist()

        scores += batch_score[:size]

        # Highest 80% are anomalous
        per = np.percentile(scores, 80)

        y_pred = scores.copy()
        y_pred = np.array(y_pred)

        inds = (y_pred < per)
        inds_comp = (y_pred >= per)

        y_pred[inds] = 0
        y_pred[inds_comp] = 1
        # 计算精度，召回率，f1分数
        precision, recall, f1, _ = precision_recall_fscore_support(testy,
                                                                   y_pred,
                                                                   average='binary')

        print(
            "Testing : Prec = %.4f | Rec = %.4f | F1 = %.4f "
            % (precision, recall, f1))


# example: 模型，bigan或者gan  dataset:数据集，kdd或minist   split: 固定为run  nb_epochs: 迭代轮次数 label: 异常标签 w: 映射损失权重 m: 鉴别器损失函数类型 d:特征匹配损失   rd:随机数种子
def run(nb_epochs, weight, method, degree, label, random_seed=42):
    """ Runs the training process"""
    """
    Python 的 with语句支持通过上下文管理器所定义的运行时上下文这一概念。 此对象的实现使用了一对专门方法，允许用户自定义类来定义运行时上下文，在语句体被执行前进入该上下文，并在语句执行完毕时退出该上下文
    """
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        train_and_test(nb_epochs, weight, method, degree, random_seed)
