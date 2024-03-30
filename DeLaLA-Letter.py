import os
import time
from collections import Counter

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import core.leading_tree as lt
import core.lmca as lm
from core.delala_select import DeLaLA_select
from utils import common


def load_parameters(param_path):
    dataset = common.load_csv(param_path)
    dc = dataset[:, 0].astype(float)
    lt_num = dataset[:, 1].astype(int)
    length_scale = dataset[:, 2].astype(float)
    return dc, lt_num, length_scale


def removeMinorData(X, Labels, k):
    """
    Remove the points and labels that have no enough neighbors
    :param X:
    :param Labels:
    :param k:
    :return:
    """
    setLabels = set(Labels)
    classNum = len(setLabels)
    cntArray = np.zeros(classNum, dtype=int)
    i = 0
    rmInds = np.zeros(0, dtype=int)
    for l in setLabels:
        clCnt = list(Labels).count(l)
        cntArray[i] = clCnt
        i += 1
        if clCnt <= k:
            lbInds = [i for i in range(len(Labels)) if Labels[i] == l]
            rmInds = np.append(rmInds, lbInds)
            # print("remove class: ", l)
    if len(rmInds) > 0:
        Remove_index = X[rmInds]
        X = np.delete(X, rmInds, axis=0)
        Labels = np.delete(Labels, rmInds)
        return X, Labels, Remove_index
    else:
        return X, Labels, -1


order = 0
param_path = os.path.join("data", "parameters.csv")
dc_all, lt_num_all, length_scale_all = load_parameters(param_path)
SelectedInds_all = np.zeros(0, dtype=int)


def PredictLabel(train_AL, label_num, layer, dimension):
    global order, SelectedInds_all
    if label_num == 0:
        pass
    elif label_num == 1:  # predict that this subtree sample has the same label as its root node
        y_test = np.delete(y[train_AL], 0)  # test set labels
        y_predict = np.zeros(len(y_test), dtype=int) + y[train_AL[0]]
        y_predict_all[train_AL] = y[train_AL[0]]
        SelectedInds_all = np.append(SelectedInds_all, train_AL[0])
        return 0
    elif 2 <= label_num <= 3 or layer == 3:  # Prediction with LMCA
        D_A = D[train_AL]
        D_A = D_A[:, train_AL]
        LT = lt.LeadingTree(X_train=X[train_AL], dc=dc_all[order], lt_num=lt_num_all[order], D=D_A)
        LT.fit()
        LTgammaPara = LT.density * LT.delta
        selectedInds = DeLaLA_select(LTgammaPara, LT.density, LT.layer, y[train_AL], 2, label_num * 2, 0.5)
        selectedInds_universe = train_AL[selectedInds]
        X_train = X[selectedInds_universe]
        y_train = y[selectedInds_universe]
        lmca = lm.LMCA(dimension=dimension, init_method="kpca", verbose=True, max_iter=100, stepsize=1.E-2,
                       nn_active=False, length_scale=length_scale_all[order], k=1)
        lmca.fit(X_train, y_train)
        order += 1
        X_test = np.delete(X[train_AL], selectedInds, axis=0)  # After removing the training set, the test set samples are obtained
        y_test = np.delete(y[train_AL], selectedInds, axis=0)  # test set labels
        y_predict = np.zeros(len(y_test), dtype=int) - 1  # predict labels

        MatDist = common.euclidian_dist_square(X_train, X_test)  # The kernel matrix corresponding to the training set and the test set
        test_bnd_K = np.exp(-1 * lmca.length_scale * MatDist).T
        B = test_bnd_K.dot(lmca.Omega)  # Test set after dimensionality reduction
        A = lmca.K.dot(lmca.Omega)  # The training set after dimensionality reduction

        # Find the training sample with the closest Euclidean distance for each test sample, and predict that both have the same label.
        D_temp = common.euclidian_dist(B, A)
        Pa = np.zeros(len(y_predict), dtype=int)  # Pa[i] represents the index of the training sample closest to the test sample i
        for j in range(len(y_predict)):
            index1 = np.argmin(D_temp[j])
            Pa[j] = index1

        y_predict = y_train[Pa]
        index_predict = np.delete(train_AL, selectedInds, axis=0)
        y_predict_all[index_predict] = y_predict
        SelectedInds_all = np.append(SelectedInds_all, selectedInds_universe)
        return 0
    else:
        return 1


if __name__ == "__main__":
    dataset_path = os.path.join("data", "letter.csv")
    X, y = common.load_data(dataset_path, label_index=0, map_label=False)

    t1 = time.time()
    scalar = MinMaxScaler()
    X = scalar.fit_transform(X)
    D = common.euclidian_dist(X, X)

    remove_index_all = np.zeros(0, dtype=int)
    y_predict_all = np.zeros(len(y), dtype=int) - 1

    dc_lt_num_arr = [[0.12, 45, 1, 2], [0.19, 10, 2, 2], [0.12, 5, 3, 8]]


    def recursive_partitioning(_X, _D, _layer=0, _train_AL=None):
        global remove_index_all
        lt1 = lt.LeadingTree(X_train=_X, dc=dc_lt_num_arr[_layer][0],
                             lt_num=dc_lt_num_arr[_layer][1], D=_D)  # Constructing the lead tree for the entire dataset
        lt1.fit()
        for i in range(dc_lt_num_arr[_layer][1]):
            if _layer == 0:
                _train_AL2, _y_AL, _remove_index = removeMinorData(lt1.AL[i], y[lt1.AL[i]], 2)
            else:
                _train_AL2, _y_AL, _remove_index = removeMinorData(_train_AL[lt1.AL[i]], y[_train_AL[lt1.AL[i]]], 2)

            _label_num = len(np.unique(y[_train_AL2]))
            remove_index_all = np.append(remove_index_all, _remove_index)
            _a = PredictLabel(_train_AL2, _label_num, dc_lt_num_arr[_layer][2], dc_lt_num_arr[_layer][3])

            if _a == 1:
                print(f"The {_layer}th layer case3: The number of subtree {i} categories is {_label_num}, which needs to be divided again.")
                _D_2 = D[_train_AL2]
                _D_2 = _D_2[:, _train_AL2]
                recursive_partitioning(X[_train_AL2], _D_2, _layer + 1, _train_AL=_train_AL2)
        return lt1


    lt0 = recursive_partitioning(X, D)

    index_None = np.zeros(0, dtype=int)  # Returns -1 if there are no indexes that need to be removed, here they are to be removed
    for i in range(len(remove_index_all)):
        if remove_index_all[i] == -1:
            index_None = np.append(index_None, i)

    remove_index_all = np.delete(remove_index_all, index_None)
    y_remove_Select = np.delete(y_predict_all, np.append(SelectedInds_all, remove_index_all))
    D = common.euclidian_dist(X[remove_index_all], X[SelectedInds_all])
    Pa = np.zeros(len(remove_index_all), dtype=int)
    for i in range(len(remove_index_all)):
        index1 = np.argmin(D[i])
        Pa[i] = index1

    y_predict_all[remove_index_all] = y[SelectedInds_all][Pa]
    arr1 = y[remove_index_all] - y_predict_all[remove_index_all]
    count0 = Counter(arr1)[0]
    print(f'The accuracy of the remove sample prediction is {count0 / len(arr1)}, {count0}/{len(arr1)}')

    # Subtree accuracy
    for i in range(45):
        temp = np.setdiff1d(lt0.AL[i], SelectedInds_all)
        arr2 = y[temp] - y_predict_all[temp]
        count1 = Counter(arr2)[0]
        print(f'The accuracy of subtree {i} is {count1 / len(temp)}, {count1}/{len(temp)}')
    # Overall accuracy
    y_predict_all = np.delete(y_predict_all, SelectedInds_all)
    y_test_all = np.delete(y, SelectedInds_all)
    arr = y_test_all - y_predict_all
    count = Counter(arr)[0]

    t2 = time.time()
    print(f'A total of {len(SelectedInds_all)} points are selected, with an accuracy of {count / len(y_test_all)}, {count}/{len(y_test_all)}')
    print(f'Takes {t2 - t1} seconds')
