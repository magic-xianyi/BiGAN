import logging
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def get_train(*args):
    """Get training dataset for KDD 10 percent"""
    return _get_adapted_dataset("train")

def get_test(*args):
    """Get testing dataset for KDD 10 percent"""
    return _get_adapted_dataset("test")

def get_shape_input():
    """Get shape of the dataset for KDD 10 percent,表示行不定，列式121"""
    return (None, 121)

def get_shape_label():
    """Get shape of the labels in KDD 10 percent"""
    return (None,)

def _get_dataset():
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 120)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 120)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    col_names = _col_names()
    df = pd.read_csv("data/kddcup.data_10_percent_corrected", header=None, names=col_names)
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    for name in text_l:
        _encode_text_dummy(df, name)  #把原始数据的字符类型替换成数值类型

    #替换label标签成数值。正常是1.错误是0
    labels = df['label'].copy()
    labels[labels != 'normal.'] = 0
    labels[labels == 'normal.'] = 1
    df['label'] = labels
    #随机抽取50%的数据，随机数种子是42，确保每次随机数生成相同。剩下50%用于test
    df_train = df.sample(frac=0.5, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)]

    #把数据转换成矩阵形式，x_train是训练数据，y_train是标签，即是否正常（0,1）
    x_train, y_train = _to_xy(df_train, target='label')
    #扁平化为一维数组，因为之前是[[0],[1],[1]]的形式
    y_train = y_train.flatten().astype(int)
    x_test, y_test = _to_xy(df_test, target='label')
    y_test = y_test.flatten().astype(int)

    #取出y_train不是1的位置的那些行数据，即非正常数据
    x_train = x_train[y_train != 1]
    y_train = y_train[y_train != 1]

    #最小最大归一化到0~1区间
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset

def _get_adapted_dataset(split):
    """ Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    dataset = _get_dataset()
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    # 如果用作测试，那么数据集处理一下，异常数据和正常数据都有。训练集只有异常数据
    if split != 'train':
        dataset[key_img], dataset[key_lbl] = _adapt(dataset[key_img],
                                                    dataset[key_lbl])

    return (dataset[key_img], dataset[key_lbl])

def _encode_text_dummy(df, name):
    """Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1]
    for red,green,blue)，即对值进行编码，因为kdd里面有些字段是字符串，不是数值，所以把他们转成数值
    """
    #df.loc用于选择数据，df.loc[行，列]
    dummies = pd.get_dummies(df.loc[:,name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df[result].values.astype(np.float32), dummies.values.astype(np.float32)

def _col_names():
    """Column names of the dataframe"""
    return ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

def _adapt(x, y, rho=0.2):
    """Adapt the ratio of normal/anomalous data， 即按一定比例返回正常、异常的数据。rho可以看做是异常值点占正常值点的比例，即rho=异常：正常"""

    # Normal data: label =0, anomalous data: label =1

    rng = np.random.RandomState(42)  # seed shuffling，随机数生成器

    inliersx = x[y == 0]
    inliersy = y[y == 0]
    outliersx = x[y == 1]  # 离群值
    outliersy = y[y == 1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)  # 生成打乱顺序的数值列
    outliersx, outliersy = outliersx[inds], outliersy[inds]  # 得到打乱顺序的异常值

    size_test = inliersx.shape[0]
    out_size_test = int(size_test*rho/(1-rho))

    outestx = outliersx[:out_size_test]  # 按比例抽取异常值点
    outesty = outliersy[:out_size_test]

    testx = np.concatenate((inliersx,outestx), axis=0)  # 拼接全部正常值和抽取的异常值点
    testy = np.concatenate((inliersy,outesty), axis=0)

    size_test = testx.shape[0]
    inds = rng.permutation(size_test)
    testx, testy = testx[inds], testy[inds]

    return testx, testy