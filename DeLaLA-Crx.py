import datetime
import os
from collections import Counter

import numpy as np

from core import leading_tree as lt, lmca as lm
from DeLaLA.delala_select2 import DeLaLA_select
from utils import common

if __name__ == "__main__":
    start_t1 = datetime.datetime.now()
    lt_num = 8  # number of subtrees
    feature_num = 15
    data_path = os.path.join("data", "crx.csv")
    X, y = common.load_data(data_path)
    X, _ = common.del_invalid(X)
    X, _, _ = common.max_min_norm(X)
    D = common.euclidian_dist(X, X)
    lt1 = lt.LeadingTree(X_train=X, dc=3, lt_num=lt_num, D=D)  # Constructing the lead tree for the entire dataset
    lt1.fit()

    LTgammaPara = lt1.density * lt1.delta
    selectedInds = DeLaLA_select(LTgammaPara, lt1.density, lt1.layer, y, 3, 6, 0.5)

    i = 0
    while i < (len(selectedInds) - 1):
        if i == 0:
            print("[", selectedInds[i], end=",")
        else:
            print("", selectedInds[i], end=",")
        i += 1
    print(selectedInds[i], "]")

    X_train = X[selectedInds]
    y_train = y[selectedInds]

    print("selectedInds: ", selectedInds)

    lmca = lm.LMCA(dimension=2, init_method="kpca", verbose=True, max_iter=100, stepsize=1.E-2,
                   nn_active=False, length_scale=0.1, k=1)
    lmca.fit(X_train, y_train)

    X_test = np.delete(X, selectedInds, axis=0)  # After removing the training set, the test set samples are obtained
    y_test = np.delete(y, selectedInds, axis=0)  # test set labels
    y_predict = np.zeros(len(y_test), dtype=int) - 1  # predict labels

    MatDist1 = common.euclidian_dist_square(X_train, X_train)
    MatDist2 = common.euclidian_dist_square(X_train, X_test)  # The kernel matrix corresponding to the training set and the test set
    test_bnd_K = np.exp(-1 * lmca.length_scale * MatDist2).T
    B = test_bnd_K.dot(lmca.Omega)  # Test set after dimensionality reduction
    A = lmca.K.dot(lmca.Omega)  # The training set after dimensionality reduction

    # y_predict = predictByLT(A, y_train, B, 0.5)
    # Find the training sample with the closest Euclidean distance for each test sample, and predict that both have the same label.
    D = common.euclidian_dist(B, A)
    Pa = np.zeros(len(y_predict), dtype=int)  # Pa[i] represents the index of the training sample closest to the test sample i
    for i in range(len(y_predict)):
        index1 = np.argmin(D[i])
        Pa[i] = index1

    y_predict = y_train[Pa]

    arr = y_predict - y_test
    count = Counter(arr)[0]
    print(f'Accuracy:{count / len(y_test)}, {count}/{len(y_test)}')
    end_t1 = datetime.datetime.now()
    elapsed_sec = (end_t1 - start_t1).total_seconds()
    print("total consumption: " + "{:.10f}".format(elapsed_sec) + "s")
