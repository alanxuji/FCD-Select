import time
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

from core import leading_tree as lt, lmca as lm
from core.delala_select import Fuzzy_select
from utils import common

if __name__ == "__main__":
    lt_num = 8  # number of subtrees
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target

    #X, _, _ = common.max_min_norm(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    DistX = common.euclidian_dist_square(X, X)
    X_length_scale = 1 / (0.7 ** 2)
    KernelX = np.exp(-1 * X_length_scale * DistX).T

    D = common.euclidian_dist(X, X)
    t1 = time.time()
    lt1 = lt.LeadingTree(X_train=X, dc=0.2, lt_num=lt_num, D=D)  # Constructing the lead tree for the entire dataset
    lt1.fit()

    X_train = np.zeros((0, 13))  # Initialize the training set
    y_train = np.zeros(0, dtype=int)  # Initialize training set labels
    index = np.zeros(0, dtype=int)  # The index corresponding to the training set

    LTgammaPara = lt1.density * lt1.delta

    selectedInds = Fuzzy_select(LTgammaPara, lt1.density, lt1.layer, y, 2, 6, 0.06)
    # selectedInds = DeLaLA_select(LTgammaPara, lt1.density, lt1.layer, y, 2, 6, 0.8)
    for i in range(len(selectedInds)):  # Add the selected samples (by continuous XOR) into KLMCA training set
        indThis = selectedInds[i]
        X_train = np.append(X_train, X[indThis].reshape(1, -1), axis=0)
        y_train = np.append(y_train, y[indThis])

    print('Layer index of the labeled samples', lt1.layer[selectedInds])
    print('index of the labeled samples', selectedInds)
    # endregion

    lmca = lm.LMCA(dimension=2, init_method="kpca", verbose=True, max_iter=100, stepsize=1.E-2,
                   nn_active=False, length_scale=1 / (0.8 ** 2),
                   k=1)  # 1.5~0.974   ###kpca is very important for the success!!!
    lmca.fit(X_train, y_train)
    print(np.shape(X_train)[0], ' Training samples.')

    X_test = np.delete(X, selectedInds, axis=0)  # After removing the training set, the test set samples are obtained
    y_test = np.delete(y, selectedInds, axis=0)  # test set labels
    y_predict = np.zeros(len(y_test), dtype=int) - 1  # predict labels

    MatDist2 = common.euclidian_dist_square(X_train, X_test)  # The kernel matrix corresponding to the training set and the test set
    test_bnd_K = np.exp(-1 * lmca.length_scale * MatDist2).T
    B = test_bnd_K.dot(lmca.Omega)  # Test set after dimensionality reduction
    A = lmca.K.dot(lmca.Omega)  # The training set after dimensionality reduction

    # Find the training sample with the closest Euclidean distance for each test sample, and predict that both have the same label.

    D = common.euclidian_dist(B, A)
    Pa = np.zeros(len(y_predict), dtype=int)  # Pa[i] represents the index of the training sample closest to the test sample i
    for i in range(len(y_predict)):
        index1 = np.argmin(D[i])
        Pa[i] = index1

    y_predict = y_train[Pa]

    arr = y_predict - y_test
    count = Counter(arr)[0]
    print("The total accuracy is", count / len(y_test), ", ", count, "/", len(y_test))

    t2 = time.time()
    print("Time elapse: ", t2 - t1)

    # region show plot
    label0 = 0
    label1 = 0
    label2 = 0

    area = np.pi * 5 ** 2
    colors = ['#00CED1', '#DC143C', '#000079', '#467500', '#613030', '#EA0000', '#84C1FF', '#8C8C00', '#FFFF37',
              '#A5A552',
              '#F00078', '#007979', '#00FFFF', '#FFD306', '#336666', '#FF00FF', '#02F78E', '#FF8000', '#5A5AAD',
              '#921AFF',
              '#6C3365', '#FF5809', '#28FF28', '#272727', '#D3D3D3', '#66CCFF']
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    fig = plt.figure(figsize=(8, 6), linewidth=2)

    ax = fig.gca()

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    plt.title("Wine", fontsize=32)
    area = np.pi * 8 ** 2
    plt.tick_params(width=2, labelsize=28)
    for i in range(len(X_train)):
        if y_train[i] == 0:
            label0 = label0 + 1
            plt.scatter(A[i, 0], A[i, 1], s=area, c=colors[0], marker='o', alpha=0.7, label='0', linewidths=2)
        if y_train[i] == 1:
            label1 = label1 + 1
            plt.scatter(A[i, 0], A[i, 1], s=area, c=colors[1], marker='^', alpha=0.7, label='1', linewidths=2)
        if y_train[i] == 2:
            label2 = label2 + 1
            plt.scatter(A[i, 0], A[i, 1], s=area, c=colors[2], marker='s', alpha=0.7, label='2', linewidths=2)
        pass
    area = np.pi * 6 ** 2
    for i in range(len(X_test)):
        if y_test[i] == 0:
            plt.scatter(B[i, 0], B[i, 1], s=area, edgecolors=colors[0], c="none", marker='o', alpha=0.4, label='0',
                        linewidths=2)
        if y_test[i] == 1:
            plt.scatter(B[i, 0], B[i, 1], s=area, edgecolors=colors[1], c="none", marker='^', alpha=0.4, label='1',
                        linewidths=2)
        if y_test[i] == 2:
            plt.scatter(B[i, 0], B[i, 1], s=area, edgecolors=colors[2], c="none", marker='s', alpha=0.4, label='2',
                        linewidths=2)
    plt.show()
