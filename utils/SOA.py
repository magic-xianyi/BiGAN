import copy
import numpy as np
from matplotlib import pyplot as plt


def initialization(pop, ub, lb, dim):
    ''' 种群初始化函数，给每个位置设置随机数值'''
    '''
    pop:为种群数量，即海鸥的个数
    dim:每个个体的维度，即变量的个数
    ub:每个维度的变量上边界，维度为[dim,1]
    lb:为每个维度的变量下边界，维度为[dim,1]
    X:为输出的种群，维度[pop,dim]
    '''
    X = np.zeros([pop, dim])  # 声明空间
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j] - lb[j]) * np.random.random() + lb[j]  # 生成[lb,ub]之间的随机数

    return X


def BorderCheck(X, ub, lb, pop, dim):
    '''边界检查函数'''
    '''
    dim:为每个个体数据的维度大小
    X:为输入数据，维度为[pop,dim]
    ub:为个体数据上边界，维度为[dim,1]
    lb:为个体数据下边界，维度为[dim,1]
    pop:为种群数量
    '''
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X

#即每个参数放到需要的函数里看，得到的值都是什么
def CaculateFitness(X, fun):
    '''计算种群的所有个体的适应度值'''
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

# 这里是升序排序，后面取第1个，而适应度函数是x^2 + y^2,所以实际上是取最小的值，可以根据需要调整适应度函数和排序函数
def SortFitness(Fit):
    '''适应度值排序'''
    '''
    输入为适应度值
    输出为排序后的适应度值，和索引
    '''
    fitness = np.sort(Fit, axis=0)   # 按列排序，axis=1表示按行排序
    index = np.argsort(Fit, axis=0)  # 排序后是下标
    return fitness, index


def SortPosition(X, index):
    '''根据适应度值对位置进行排序'''
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def SOA(pop, dim, lb, ub, MaxIter, fun):
    '''海鸥优化算法'''
    '''
    输入：
    pop:为种群数量
    dim:每个个体的维度
    ub:为个体上边界信息，维度为[1,dim]
    lb:为个体下边界信息，维度为[1,dim]
    fun:为适应度函数接口
    MaxIter:为最大迭代次数
    输出：
    GbestScore:最优解对应的适应度值
    GbestPositon:最优解
    Curve:迭代曲线
    '''
    fc = 2  # 可调
    X = initialization(pop, ub, lb, dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值，fitness形状：[pop, 1]
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序，fitness是排序后的值表，sortIndex是排序后的下标表
    X = SortPosition(X, sortIndex)  # 种群排序，根据适应度函数的值排序每只海鸥的位置，得到最佳海鸥，即X[0],其各维度的解就是最好的解
    GbestScore = copy.copy(fitness[0])  # 存储最好得分
    GbestPositon = np.zeros([1, dim])  # 存储最好海鸥的值，即最好的各个变量的解
    GbestPositon[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([MaxIter, 1])
    MS = np.zeros([pop, dim])
    CS = np.zeros([pop, dim])
    DS = np.zeros([pop, dim])
    X_new = copy.copy(X)
    for i in range(MaxIter):
        print("第" + str(i) + "次迭代")
        Pbest = X[0, :]
        for j in range(pop):
            # 计算Cs
            A = fc - (i * (fc / MaxIter))
            CS[j, :] = X[j, :] * A
            # 计算Ms
            rd = np.random.random()
            B = 2 * (A ** 2) * rd
            MS[j, :] = B * (Pbest - X[j, :])
            # 计算Ds
            DS[j, :] = np.abs(CS[j, :] + MS[j, :])
            # 局部搜索
            u = 1
            v = 1
            theta = np.random.random()
            r = u * np.exp(theta * v)
            x = r * np.cos(theta * 2 * np.pi)
            y = r * np.sin(theta * 2 * np.pi)
            z = r * theta
            # 攻击
            X_new[j, :] = x * y * z * DS[j, :] + Pbest
        X = BorderCheck(X_new, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        """
        这里其实就是解，GbestPositon就是我们需要的解向量，各列代表各个变量的解，GbestScore为响应场景下适应度函数的得分，我们根据GbestPositon拿出各个变量的值即可
        """
        if (fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[i] = GbestScore

    return GbestScore, GbestPositon, Curve


'''适应度函数'''
def fun(X):
    O = X[0] ** 2 + X[1] ** 2
    return O


'''海鸥优化算法求解x1^2 + x2^2的最小值'''
'''主函数 '''
if __name__ == '__main__':
    # 设置参数
    pop = 50  # 种群数量
    MaxIter = 30  # 最大迭代次数
    dim = 2  # 维度
    lb = -10 * np.ones(dim)  # 下边界
    ub = 10 * np.ones(dim)  # 上边界
    # 适应度函数选择
    fobj = fun
    GbestScore, GbestPositon, Curve = SOA(pop, dim, lb, ub, MaxIter, fobj)
    print('最优适应度值：', GbestScore)
    print('最优解[x1,x2]：', GbestPositon)

    # 绘制适应度曲线
    plt.figure(1)
    plt.plot(Curve, 'r-', linewidth=2)
    plt.xlabel('Iteration', fontsize='medium')
    plt.ylabel("Fitness", fontsize='medium')
    plt.grid()
    plt.title('SOA', fontsize='large')
    plt.show()