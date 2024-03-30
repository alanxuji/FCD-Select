import os
from collections import Counter

import numpy as np
from core import leading_tree as lt, lmca as lm
from core.delala_select import DeLaLA_select
from sklearn.preprocessing import Normalizer
import time
import scipy.io as scio
from utils import common


def EuclidianDist2(X1, X2):
    ###Using broadcasting, simpler and faster!
    tempM = np.sum(X1 ** 2, 1).reshape(-1, 1)  ##行数不知道，只知道列数为1
    tempN = np.sum(X2 ** 2, 1)  # X2 ** 2: element-wise square, sum(_,1): 沿行方向相加，但最后是得到行向量
    sqdist = tempM + tempN - 2 * np.dot(X1, X2.T)
    sqdist[sqdist < 0] = 0
    return np.sqrt(sqdist)


def EuclidianDistsq(X1, X2):
    ###Using broadcasting, simpler and faster!
    tempM = np.sum(X1 ** 2, 1).reshape(-1, 1)  ##行数不知道，只知道列数为1
    tempN = np.sum(X2 ** 2, 1)  # X2 ** 2: element-wise square, sum(_,1): 沿行方向相加，但最后是得到行向量
    sqdist = tempM + tempN - 2 * np.dot(X1, X2.T)
    sqdist[sqdist < 0] = 0
    return sqdist


def findUnshownClass(y_train, labels, layers, rho, k, n_addPerClass):
    """
    :param y_train: labels of the selected data points
    :param Y: All labels
    :param rho: local density of each sample
    :param n_addPerClass: how many nodes to add per unshown class
    :param layers: layer Index of each data point in the subtrees
    :return: Indics of the samples to add
    """
    y_unique = set(y_train)
    # cntArray = np.bincount(y_train) ###this needs change

    classNum = len(y_unique)
    cntArray = np.zeros(classNum, dtype=int)
    i = 0
    for l in y_unique:
        cntArray[i] = list(y_train).count(l)
        i += 1

    rareInds = [labels[i] for i in range(len(cntArray)) if cntArray[i] < k + 1]
    AllLabels = set(labels)
    UnshownClasses = AllLabels.difference(y_unique)
    UnshownClasses = UnshownClasses.union(set(rareInds))
    popularNum = int(np.ceil(n_addPerClass / 2))
    devergentNum = int(n_addPerClass - popularNum)
    XInd_Add = np.zeros(0, dtype=int)
    for cl in UnshownClasses:
        iClassInds = np.array([i for i in range(len(labels)) if labels[i] == cl])

        ### add according to local density
        classRho = rho[iClassInds]
        sortIndsRho = np.argsort(classRho)
        temIndsR = sortIndsRho[:popularNum]
        Xpop = iClassInds[temIndsR]

        ### add according to layer index
        classLayer = layers[iClassInds]
        sortIndsLayer = np.argsort(classLayer)
        temIndsD = sortIndsLayer[:devergentNum]
        XDevergent = iClassInds[temIndsD]

        XInd_Add = np.append(XInd_Add, Xpop)
        XInd_Add = np.append(XInd_Add, XDevergent)
    return XInd_Add


def MapLabels(y, y_train):
    """
    Map the labels to follow the [0,1,...,K-1] label protocol
    :param y:
    :param y_train:
    :return:
    """
    ySet = set(y_train)
    yNotInTrain = set(y).difference(ySet)

    for lnoShow in yNotInTrain:
        for i in range(len(y)):
            if y[i] == lnoShow:
                y[i] = -1
    ind = 0
    for l in ySet:
        for j in range(len(y_train)):
            if y_train[j] == l:
                y_train[j] = ind
        for j in range(len(y)):
            if y[j] == l:
                y[j] = ind
        ind += 1
    print("y:", y)
    print("y_train:", y_train)
    return y, y_train


def removeMinorData(X, Labels, k):
    """
    Remove the points and labels that have no enough neighbors
    :param X:
    :param Labels:
    :param k:
    :return:
    """
    # setLabels = set(Labels)
    setLabels = np.unique(Labels)
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
            print("remove class: ", l)
    if len(rmInds) > 0:
        print(f'移除{len(rmInds)}个样本')
        Remove_index = X[rmInds]
        X = np.delete(X, rmInds, axis=0)
        Labels = np.delete(Labels, rmInds)
        return X, Labels, Remove_index
    else:
        return X, Labels, -1


lt_num = 8  # 子树个数
k = 2
data = scio.loadmat(os.path.join("data", "ORL_32x32.mat"))
X, y = np.array(data["fea"]), np.array(data["gnd"])
y = y.flatten()
y = y-1
dim = X.shape[1]

t1 = time.time()
# scalar = MinMaxScaler()
# X = scalar.fit_transform(X)
Normalizer = Normalizer()
X = Normalizer.fit_transform(X)
t3 = time.time()
D = common.euclidian_dist(X, X)
#lt1 = lt.LeadingTree(X_train=X, dc=0.24, lt_num=lt_num, D= )  # 整个数据集构造引领树
lt1 = lt.LeadingTree(X_train=X, dc=0.24, lt_num=lt_num, D=D )
lt1.fit()

### Here split several subtree to train local metrics and check wether all classes are covered
# X_train = np.zeros((0, dim))  # 初始化训练集为
# y_train = np.zeros(0, dtype=int)  # 初始化训练集标签
#
# datasetX = np.zeros((0, dim))  # sub dataset consists of several leading trees
# datasetY = np.zeros(0, dtype=int)
#
# # 训练集对应的索引
# SubsetIndex = np.zeros(0, dtype=int)  ##sub dataset Index
# totalCNT = 0
# for ltGroup in range(lt_num):
#     i = ltGroup
#     index = np.zeros(0, dtype=int)
#     datasetX = np.zeros((0, dim))  # sub dataset consists of several leading trees
#     datasetY = np.zeros(0, dtype=int)
#     index = np.append(index, lt1.AL[i][0])
#
#     X_train = np.append(X_train, X[lt1.AL[i][0]].reshape(1, -1), axis=0)
#     y_train = np.append(y_train, y[lt1.AL[i][0]])
#     SubsetIndex = np.append(SubsetIndex, lt1.AL[i])
#     for j in lt1.AL[i]:  ### extract sub dataset
#         datasetX = np.append(datasetX, X[j].reshape(1, -1), axis=0)
#         datasetY = np.append(datasetY, y[j])
#     plt.subplot(5, 9, ltGroup + 1)
#     plt.hist(datasetY, bins=26, rwidth=2)
#     ltPopu = len(datasetY)
#     class_num = len(np.unique(datasetY))
#     print("LT #", i, ": Population: ", ltPopu, " Class num: ", class_num)
#     totalCNT += ltPopu
#
# # if i == 9:
# #     with open("datasets/LT9.csv", "w", encoding="gbk", newline="") as f:
# #         csv_writer = csv.writer(f)
# #         for j in range(ltPopu):
# #             csv_writer.writerow(np.append(datasetY[j], datasetX[j, :]))
# #         print("写入数据成功")
# #         f.close()
#
# plt.show()
# # print(totalCNT, " nodes in trees.")

remove_index_all = np.zeros(0, dtype=int)
y_predict_all = np.zeros(len(y), dtype=int) - 1
SelectedInds_all = np.zeros(0, dtype=int)
#
# #  参数设置
#
#  第一层

#  第二层

for i in range(lt_num):
    train_AL, y_AL, remove_index = removeMinorData(lt1.AL[i], y[lt1.AL[i]], 2)
    label_num = len(np.unique(y[train_AL]))
    remove_index_all = np.append(remove_index_all, remove_index)
    if label_num == 0:
        pass
    elif label_num == 1:  # 预测该子树样本与其根节点标签相同
        y_test = np.delete(y[train_AL], 0)  # 测试集标签
        y_predict = np.zeros(len(y_test), dtype=int) + y[train_AL[0]]
        arr = y_test - y_predict
        count = Counter(arr)[0]
        print(f'第一层case1:子树{i}的准确率为{count / len(y_test)}, {count}/{len(y_test)}')

        y_predict_all[train_AL] = y[train_AL[0]]
        SelectedInds_all = np.append(SelectedInds_all, train_AL[0])
    elif 2 <= label_num <= 3:  # 用LMCA预测
        DAL = D = common.euclidian_dist(X[train_AL], X[train_AL])
        LT = lt.LeadingTree(X_train=X[train_AL], dc=3, lt_num=3, D=DAL)
        LT.fit()
        LTgammaPara = LT.density * LT.delta
        selectedInds = DeLaLA_select(LTgammaPara, LT.density, LT.layer, y[train_AL], 2, label_num * 2, 0.5)
        selectedInds_universe = train_AL[selectedInds]
        X_train = X[selectedInds_universe]
        y_train = y[selectedInds_universe]
        lmca = lm.LMCA(dimension=50, init_method="kpca", verbose=True, max_iter=100, stepsize=1.E-2,
                       nn_active=False, length_scale=0.1, k=1)
        lmca.fit(X_train, y_train)
        X_test = np.delete(X[train_AL], selectedInds, axis=0)  # 去除训练集后得到测试集样本
        y_test = np.delete(y[train_AL], selectedInds, axis=0)  # 测试集标签
        y_predict = np.zeros(len(y_test), dtype=int) - 1  # 预测标签

        MatDist = EuclidianDistsq(X_train, X_test)  # 训练集和测试集对应的核矩阵
        test_bnd_K = np.exp(-1 * lmca.length_scale * MatDist).T
        B = test_bnd_K.dot(lmca.Omega)  # 降维后的测试集
        A = lmca.K.dot(lmca.Omega)  # 降维后的训练集

        # 为每个测试样本找到欧氏距离最近的训练样本，预测二者标签相同。
        D_tt = EuclidianDist2(B, A)
        Pa = np.zeros(len(y_predict), dtype=int)  # Pa[i]代表距离测试样本i最近的训练样本的索引
        for j in range(len(y_predict)):
            index1 = np.argmin(D_tt[j])
            Pa[j] = index1

        y_predict = y_train[Pa]
        arr = y_test - y_predict
        count = Counter(arr)[0]
        print(f'第一层case2:子树{i}类别数为{len(np.unique(y[train_AL]))}, 准确率为{count / len(y_test)}, {count}/{len(y_test)}')

        index_predict = np.delete(train_AL, selectedInds, axis=0)
        y_predict_all[index_predict] = y_predict
        SelectedInds_all = np.append(SelectedInds_all, selectedInds_universe)
    else:
        print(f"第一层case3:子树{i}类别数为{label_num},需再次划分")
        DAL = D = common.euclidian_dist(X[train_AL], X[train_AL])
        lt2 = lt.LeadingTree(X_train=X[train_AL], dc=0.2, lt_num=2, D=DAL )  # lt2.AL是相对索引
        lt2.fit()
        # order += 1
        for j in range(lt2.lt_num):
            train_AL_2, y_AL_2, remove_index_2 = removeMinorData(train_AL[lt2.AL[j]], y[train_AL[lt2.AL[j]]],
                                                                 2)  # train_AL是绝对索引
            label_num = len(np.unique(y[train_AL_2]))
            remove_index_all = np.append(remove_index_all, remove_index_2)
            print(f'子树{i}-{j}类别数为{label_num}, 总数为{len(train_AL_2)}')
            if label_num == 0:
                pass
            elif label_num == 1:  # 预测该子树样本与其根节点标签相同
                y_test = np.delete(y[train_AL_2], 0)  # 测试集标签
                y_predict = np.zeros(len(y_test), dtype=int) + y[train_AL_2[0]]
                arr = y_test - y_predict
                count = Counter(arr)[0]
                print(f'第二层case1:子树{i}-{j}的准确率为{count / len(y_test)}, {count}/{len(y_test)}')

                y_predict_all[train_AL_2] = y[train_AL_2[0]]
                SelectedInds_all = np.append(SelectedInds_all, train_AL_2[0])

            elif 2 <= label_num <= 3:  # 用LMCA预测
                # lt_num = ['lt_num_' + str(i) + '_' + str(j)][order_2]
                DAL2  = common.euclidian_dist(X[train_AL_2], X[train_AL_2])
                LT = lt.LeadingTree(X_train=X[train_AL_2], dc=0.1, lt_num=3, D =DAL2)
                LT.fit()
                LTgammaPara = LT.density * LT.delta
                selectedInds = DeLaLA_select(LTgammaPara, LT.density, LT.layer, y[train_AL_2], 2, label_num * 2,
                                             0.5)  # selectedInds是相对索引
                selectedInds_universe = train_AL_2[selectedInds]
                X_train = X[selectedInds_universe]
                y_train = y[selectedInds_universe]
                lmca = lm.LMCA(dimension=50, init_method="kpca", verbose=True, max_iter=100, stepsize=1.E-2,
                               nn_active=False, length_scale=0.1, k=1)
                lmca.fit(X_train, y_train)
                X_test = np.delete(X[train_AL_2], selectedInds, axis=0)  # 去除训练集后得到测试集样本
                y_test = np.delete(y[train_AL_2], selectedInds, axis=0)  # 测试集标签
                y_predict = np.zeros(len(y_test), dtype=int) - 1  # 预测标签

                MatDist = EuclidianDistsq(X_train, X_test)  # 训练集和测试集对应的核矩阵
                test_bnd_K = np.exp(-1 * lmca.length_scale * MatDist).T
                B = test_bnd_K.dot(lmca.Omega)  # 降维后的测试集
                A = lmca.K.dot(lmca.Omega)  # 降维后的训练集

                # 为每个测试样本找到欧氏距离最近的训练样本，预测二者标签相同。
                D_tt = EuclidianDist2(B, A)
                Pa = np.zeros(len(y_predict), dtype=int)  # Pa[i]代表距离测试样本i最近的训练样本的索引
                for n in range(len(y_predict)):
                    index1 = np.argmin(D_tt[n])
                    Pa[n] = index1

                y_predict = y_train[Pa]
                arr = y_test - y_predict
                count = Counter(arr)[0]
                print(
                    f'第二层case2:子树{i}-{j}类别数为{len(np.unique(y[train_AL_2]))}, 准确率为{count / len(y_test)}, {count}/{len(y_test)}')

                index_predict = np.delete(train_AL_2, selectedInds, axis=0)
                y_predict_all[index_predict] = y_predict
                SelectedInds_all = np.append(SelectedInds_all, selectedInds_universe)
            else:
                print(f"case3:子树{i}类别数为{label_num},需二次划分")
                DAL2 = D = common.euclidian_dist(X[train_AL_2], X[train_AL_2])
                lt3 = lt.LeadingTree(X_train=X[train_AL_2], dc=0.12, lt_num=3, D= DAL2 )  # lt2.AL是相对索引
                lt3.fit()
                for n in range(lt3.lt_num):
                    train_AL_3, y_AL_3, remove_index_3 = removeMinorData(train_AL_2[lt3.AL[n]],
                                                                         y[train_AL_2[lt3.AL[n]]], 2)  # train_AL是绝对索引
                    label_num = len(np.unique(y[train_AL_3]))
                    remove_index_all = np.append(remove_index_all, remove_index_3)
                    # print(f'子树{i}二次划分后的子树{n}类别数为{label_num}, 总数为{len(train_AL_3)}')
                    if label_num == 0:
                        pass
                    elif label_num == 1:  # 预测该子树样本与其根节点标签相同
                        y_test = np.delete(y[train_AL_3], 0)  # 测试集标签
                        y_predict = np.zeros(len(y_test), dtype=int) + y[train_AL_3[0]]
                        arr = y_test - y_predict
                        count = Counter(arr)[0]
                        print(f'第三层case1:子树{i}-{j}-{n}的准确率为{count / len(y_test)}, {count}/{len(y_test)}')
                        y_predict_all[train_AL_3] = y[train_AL_3[0]]
                        SelectedInds_all = np.append(SelectedInds_all, train_AL_3[0])
                    else:
                        DAL3 = D = common.euclidian_dist(X[train_AL_3], X[train_AL_3])
                        LT = lt.LeadingTree(X_train=X[train_AL_3], dc=0.12, lt_num=3, D=DAL3)
                        LT.fit()
                        LTgammaPara = LT.density * LT.delta
                        selectedInds = DeLaLA_select(LTgammaPara, LT.density, LT.layer, y[train_AL_3], 2,
                                                     label_num * 2,
                                                     0.5)  # selectedInds是相对索引
                        selectedInds_universe = train_AL_3[selectedInds]
                        X_train = X[selectedInds_universe]
                        y_train = y[selectedInds_universe]
                        lmca = lm.LMCA(dimension=50, init_method="kpca", verbose=True, max_iter=100, stepsize=1.E-2,
                                       nn_active=False, length_scale=0.1, k=1)
                        lmca.fit(X_train, y_train)

                        X_test = np.delete(X[train_AL_3], selectedInds, axis=0)  # 去除训练集后得到测试集样本
                        y_test = np.delete(y[train_AL_3], selectedInds, axis=0)  # 测试集标签
                        y_predict = np.zeros(len(y_test), dtype=int) - 1  # 预测标签

                        MatDist = EuclidianDistsq(X_train, X_test)  # 训练集和测试集对应的核矩阵
                        test_bnd_K = np.exp(-1 * lmca.length_scale * MatDist).T
                        B = test_bnd_K.dot(lmca.Omega)  # 降维后的测试集
                        A = lmca.K.dot(lmca.Omega)  # 降维后的训练集

                        # 为每个测试样本找到欧氏距离最近的训练样本，预测二者标签相同。
                        D_tt = EuclidianDist2(B, A)
                        Pa = np.zeros(len(y_predict), dtype=int)  # Pa[i]代表距离测试样本i最近的训练样本的索引
                        for m in range(len(y_predict)):
                            index1 = np.argmin(D_tt[m])
                            Pa[m] = index1

                        y_predict = y_train[Pa]
                        arr = y_test - y_predict
                        count = Counter(arr)[0]
                        print(
                            f'第三层case2:子树{i}-{j}-{n}类别数为{len(np.unique(y[train_AL_3]))}, 准确率为{count / len(y_test)}, {count}/{len(y_test)}')

                        index_predict = np.delete(train_AL_3, selectedInds, axis=0)
                        y_predict_all[index_predict] = y_predict
                        SelectedInds_all = np.append(SelectedInds_all, selectedInds_universe)

y_predict_all[SelectedInds_all] = y[SelectedInds_all]

index_None = np.zeros(0, dtype=int)  # 如果没有需要remove的索引则会返回-1，在这里要删除它们
for i in range(len(remove_index_all)):
    if remove_index_all[i] == -1:
        index_None = np.append(index_None, i)

# remove_index_all = np.delete(remove_index_all, index_None)
# y_Select = y[SelectedInds_all]
# D = EuclidianDist2(X[remove_index_all], X[SelectedInds_all])
# Pa = np.zeros(len(remove_index_all), dtype=int)
# for i in range(len(remove_index_all)):
#     index1 = np.argmin(D[i])
#     Pa[i] = index1
# y_predict_all[remove_index_all] = y_Select[Pa]
# arr1 = y[remove_index_all] - y_predict_all[remove_index_all]
# count0 = Counter(arr1)[0]
# print(f'被移除样本预测的准确率为{count0 / len(arr1)}, {count0}/{len(arr1)}')

# remove_index_all = np.delete(remove_index_all, index_None)
# y_remove_Select = np.delete(y_predict_all, remove_index_all)
# D = EuclidianDist2(X[remove_index_all], np.delete(X, remove_index_all, axis=0))
# Pa = np.zeros(len(remove_index_all), dtype=int)
# for i in range(len(remove_index_all)):
#     index1 = np.argmin(D[i])
#     Pa[i] = index1
# y_predict_all[remove_index_all] = y_remove_Select[Pa]
# arr1 = y[remove_index_all] - y_predict_all[remove_index_all]
# count0 = Counter(arr1)[0]
# print(f'被移除样本预测的准确率为{count0 / len(arr1)}, {count0}/{len(arr1)}')

# knn
# remove_index_all = np.delete(remove_index_all, index_None)
# knn = KNeighborsRegressor(n_neighbors=1)
# X_train = np.delete(X, remove_index_all, axis=0)
# y_train = np.delete(y, remove_index_all)
# knn.fit(X_train, y_train)
# y_remove = knn.predict(X[remove_index_all])
# arr1 = y[remove_index_all] - y_remove
# count0 = Counter(arr1)[0]
# print(f'被移除样本预测的准确率为{count0 / len(arr1)}, {count0}/{len(arr1)}')

remove_index_all = np.delete(remove_index_all, index_None)
train_AL_4, y_AL_4, remove_index_4 = removeMinorData(remove_index_all, y[remove_index_all], 2)
SelectedInds_all = np.append(SelectedInds_all, remove_index_4)
y_predict_all[remove_index_4] = y[remove_index_4]
label_num = len(np.unique(y_AL_4))
DAL4 = D = common.euclidian_dist(X[train_AL_4], X[train_AL_4])
LT = lt.LeadingTree(X_train=X[train_AL_4], dc=3, lt_num=3, D = DAL4 )
LT.fit()
LTgammaPara = LT.density * LT.delta
selectedInds = DeLaLA_select(LTgammaPara, LT.density, LT.layer, y[train_AL_4], 2, label_num * 2, 0.5)  # selectedInds是相对索引
selectedInds_universe = train_AL_4[selectedInds]
a = np.intersect1d(SelectedInds_all, selectedInds_universe)
X_train = X[selectedInds_universe]
y_train = y[selectedInds_universe]
lmca = lm.LMCA(dimension=50, init_method="kpca", verbose=True, max_iter=100, stepsize=1.E-2,
               nn_active=False, length_scale=3, k=1)
lmca.fit(X_train, y_train)

X_test = np.delete(X[train_AL_4], selectedInds, axis=0)  # 去除训练集后得到测试集样本
y_test = np.delete(y[train_AL_4], selectedInds, axis=0)  # 测试集标签
y_predict = np.zeros(len(y_test), dtype=int) - 1  # 预测标签

MatDist = EuclidianDistsq(X_train, X_test)  # 训练集和测试集对应的核矩阵
test_bnd_K = np.exp(-1 * lmca.length_scale * MatDist).T
B = test_bnd_K.dot(lmca.Omega)  # 降维后的测试集
A = lmca.K.dot(lmca.Omega)  # 降维后的训练集

# 为每个测试样本找到欧氏距离最近的训练样本，预测二者标签相同。
D_tt = EuclidianDist2(B, A)
Pa = np.zeros(len(y_predict), dtype=int)  # Pa[i]代表距离测试样本i最近的训练样本的索引
for m in range(len(y_predict)):
    index1 = np.argmin(D_tt[m])
    Pa[m] = index1

y_predict = y_train[Pa]
arr = y_test - y_predict
count = Counter(arr)[0]
print(f'{count}/{len(y_test)}')

index_predict = np.delete(train_AL_4, selectedInds, axis=0)
y_predict_all[index_predict] = y_predict
SelectedInds_all = np.append(SelectedInds_all, selectedInds_universe)

# remove_index_all = np.delete(remove_index_all, index_None)
# y_remove = y[remove_index_all]
# c = np.unique(y[remove_index_all])
# for i in c:
#     count = Counter(y_remove)[i]
#     print(f'类别{i}的样本数为{count}')


# 总准确率
y_predict_all = np.delete(y_predict_all, SelectedInds_all)
y_test_all = np.delete(y, SelectedInds_all)
arr = y_test_all - y_predict_all
count = Counter(arr)[0]

t2 = time.time()
print(f'共选择{len(SelectedInds_all)}个点, 准确率为{count / len(y_test_all)}, {count}/{len(y_test_all)}')
print(f'距离矩阵计算耗时{t3 - t1}秒')
print(f'耗时{t2 - t1}秒')
