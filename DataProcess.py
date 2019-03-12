import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import pylab as plt

train_file = 'training.csv'
test_file = 'test.csv'


def load(test=False, cols=False):
    """
    载入数据，通过参数控制载入训练集还是测试集，并筛选特征列
    """
    fname = test_file if test else train_file
    df = pd.read_csv(os.path.expanduser(fname))

    # 将图像数据转换为数组
    df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, sep=' '))

    # 筛选指定的数据列
    if cols:
        cols = df.keys()
        df = df[cols]

    # print(df.count())  # 每列的简单统计
    df = df.dropna()  # 删除空数据

    # 归一化到0到1
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)

    # 针对训练集目标标签进行归一化
    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


# 将单行像素数据转换为三维矩阵
def load2d(test=False, cols=True):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 96, 96, 1)
    return X, y


if __name__ == "__main__":
    X, y = load2d(test=False, cols=True)
    plt.imshow(X[0].reshape(96,96))
    plt.show()
    print("TrainingData:");
    print(X[0])
    print("---------------------------------")
    print(y[0])
    print(len(X), len(y))
    X, y = load(test=True, cols=False)
    print("TestData:")
    print(X[0])
    print("---------------------------------")
    assert y == None
    print(len(X))
