from collections import Counter
from matplotlib import pyplot as plt
import MyLeadingTreeG as lt
import numpy as np
from load_parameters import load_parameters
import LMCA_Mine as lm
from DeLaLA_select import DeLaLA_select
from sklearn.preprocessing import MinMaxScaler
import time
from load_yeast import load_yeast


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
            print("remove class: ", l)
    if len(rmInds) > 0:
        Remove_index = X[rmInds]
        X = np.delete(X, rmInds, axis=0)
        Labels = np.delete(Labels, rmInds)
        return X, Labels, Remove_index
    else:
        return X, Labels, -1


X, y = load_yeast()
lt_num = 8
dim = X.shape[1]
t1 = time.time()
scalar = MinMaxScaler()
X = scalar.fit_transform(X)
# Normalizer = Normalizer()
# X = Normalizer.fit_transform(X)
print('开始计算引领树')
lt1 = lt.LeadingTree(X_train=X, dc=0.08, lt_num=lt_num)  # 整个数据集构造引领树
lt1.fit()
t2 = time.time()
print('引领树计算完毕,耗时', (t2 - t1))

# Here split several subtree to train local metrics and check wether all classes are covered
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
#     # plt.subplot(5, 6, ltGroup + 1)
#     # plt.hist(datasetY, bins=26, rwidth=2)
#     ltPopu = len(datasetY)
#     class_num = len(np.unique(datasetY))
#     print("LT #", i, ": Population: ", ltPopu, " Class num: ", class_num)
#     totalCNT += ltPopu
#
#     # if i == 7:
#     #     with open("datasets/LT7.csv", "w", encoding="gbk", newline="") as f:
#     #         csv_writer = csv.writer(f)
#     #         for j in range(ltPopu):
#     #             csv_writer.writerow(np.append(datasetY[j], datasetX[j, :]))
#     #         print("写入数据成功")
#     #         f.close()
#
# # plt.show()
# print(totalCNT, " nodes in trees.")
#
remove_index_all = np.zeros(0, dtype=int)
y_predict_all = np.zeros(len(y), dtype=int) - 1
SelectedInds_all = np.zeros(0, dtype=int)
#
# #  参数设置
#
# #  第一层
order = 0
dc_1 = np.zeros(2) + 0.01
dc_1[1] = 0.1

lt_num_1 = np.zeros(2, dtype=int) + 2

length_scale_1 = np.zeros(2) + 0.5

#
# #  第三层
order3 = 0
dc_3, lt_num_3, length_scale_3 = load_parameters()
#
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
        SelectedInds_all = np.append(SelectedInds_all, lt1.AL[i][0])
    elif 2 <= label_num <= 3:  # 用LMCA预测
        LT = lt.LeadingTree(X_train=X[train_AL], dc=dc_1[order], lt_num=lt_num_1[order])
        LT.fit()
        selectedInds = DeLaLA_select(LT.gamma, LT.density, LT.layer, y[train_AL], 2, label_num * 2, 0.5)
        selectedInds_universe = train_AL[selectedInds]
        X_train = X[selectedInds_universe]
        y_train = y[selectedInds_universe]
        lmca = lm.LMCA(dimension=2, init_method="kpca", verbose=True, max_iter=100, stepsize=1.E-2,
                       nn_active=False, length_scale=length_scale_1[order], k=1)
        lmca.fit(X_train, y_train)
        order = order + 1
        X_test = np.delete(X[train_AL], selectedInds, axis=0)  # 去除训练集后得到测试集样本
        y_test = np.delete(y[train_AL], selectedInds, axis=0)  # 测试集标签
        y_predict = np.zeros(len(y_test), dtype=int) - 1  # 预测标签

        MatDist = EuclidianDistsq(X_train, X_test)  # 训练集和测试集对应的核矩阵
        test_bnd_K = np.exp(-1 * lmca.length_scale * MatDist).T
        B = test_bnd_K.dot(lmca.Omega)  # 降维后的测试集
        A = lmca.K.dot(lmca.Omega)  # 降维后的训练集

        # 为每个测试样本找到欧氏距离最近的训练样本，预测二者标签相同。
        D = EuclidianDist2(B, A)
        Pa = np.zeros(len(y_predict), dtype=int)  # Pa[i]代表距离测试样本i最近的训练样本的索引
        for j in range(len(y_predict)):
            index1 = np.argmin(D[j])
            Pa[j] = index1

        y_predict = y_train[Pa]
        arr = y_test - y_predict
        count = Counter(arr)[0]
        print(f'第一层case2:子树{i}类别数为{len(np.unique(y[train_AL]))}, 准确率为{count / len(y_test)}, {count}/{len(y_test)}'
              f'*****dc={LT.dc}, lt_num={LT.lt_num}, length_scale={lmca.length_scale}')

        index_predict = np.delete(train_AL, selectedInds, axis=0)
        y_predict_all[index_predict] = y_predict
        SelectedInds_all = np.append(SelectedInds_all, selectedInds_universe)
    else:
        print(f"第一层case3:子树{i}类别数为{label_num},需再次划分")
        lt2 = lt.LeadingTree(X_train=X[train_AL], dc=1.5, lt_num=3)
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
                print(f'第二层case1:子树{i}再划分后的子树{j}的准确率为{count / len(y_test)}, {count}/{len(y_test)}')

                # y_predict_all[train_AL[lt2.AL[j]]] = y[train_AL_2[0]]
                # SelectedInds_all = np.append(SelectedInds_all, train_AL[lt2.AL[j][0]])
                y_predict_all[train_AL_2] = y[train_AL_2[0]]
                SelectedInds_all = np.append(SelectedInds_all, train_AL_2[0])

            elif 2 <= label_num <= 3:  # 用LMCA预测
                LT = lt.LeadingTree(X_train=X[train_AL_2], dc=0.01, lt_num=3)
                # LT = lt.LeadingTree(X_train=X[train_AL_2], dc=1.5, lt_num=3)
                LT.fit()
                LTgammaPara = LT.density * LT.delta
                selectedInds = DeLaLA_select(LTgammaPara, LT.density, LT.layer, y[train_AL_2], 2, label_num * 2,
                                             0.5)  # selectedInds是相对索引
                selectedInds_universe = train_AL_2[selectedInds]
                X_train = X[selectedInds_universe]
                y_train = y[selectedInds_universe]
                lmca = lm.LMCA(dimension=2, init_method="kpca", verbose=True, max_iter=100, stepsize=1.E-2,
                               nn_active=False, length_scale=0.01, k=1)
                lmca.fit(X_train, y_train)
                X_test = np.delete(X[train_AL_2], selectedInds, axis=0)  # 去除训练集后得到测试集样本
                y_test = np.delete(y[train_AL_2], selectedInds, axis=0)  # 测试集标签
                y_predict = np.zeros(len(y_test), dtype=int) - 1  # 预测标签

                MatDist = EuclidianDistsq(X_train, X_test)  # 训练集和测试集对应的核矩阵
                test_bnd_K = np.exp(-1 * lmca.length_scale * MatDist).T
                B = test_bnd_K.dot(lmca.Omega)  # 降维后的测试集
                A = lmca.K.dot(lmca.Omega)  # 降维后的训练集

                # 为每个测试样本找到欧氏距离最近的训练样本，预测二者标签相同。
                D = EuclidianDist2(B, A)
                Pa = np.zeros(len(y_predict), dtype=int)  # Pa[i]代表距离测试样本i最近的训练样本的索引
                for n in range(len(y_predict)):
                    index1 = np.argmin(D[n])
                    Pa[n] = index1

                y_predict = y_train[Pa]
                arr = y_test - y_predict
                count = Counter(arr)[0]
                print(
                    f'第二层case2:子树{i}-{j}类别数为{len(np.unique(y[train_AL_2]))}, 准确率为{count / len(y_test)}, {count}/{len(y_test)}'
                    f'*****dc={LT.dc}, lt_num={LT.lt_num}, length_scale={lmca.length_scale}')

                index_predict = np.delete(train_AL_2, selectedInds, axis=0)
                y_predict_all[index_predict] = y_predict
                SelectedInds_all = np.append(SelectedInds_all, selectedInds_universe)
            else:
                print(f"第二层case3:子树{i}类别数为{label_num},需二次划分")
                lt3 = lt.LeadingTree(X_train=X[train_AL_2], dc=0.02, lt_num=10)  # lt2.AL是相对索引
                lt3.fit()
                # order += 1
                for n in range(lt3.lt_num):
                    train_AL_3, y_AL_3, remove_index_3 = removeMinorData(train_AL_2[lt3.AL[n]],
                                                                         y[train_AL_2[lt3.AL[n]]],
                                                                         2)  # train_AL是绝对索引
                    label_num = len(np.unique(y[train_AL_3]))
                    remove_index_all = np.append(remove_index_all, remove_index_3)
                    print(f'子树{i}-{j}-{n}类别数为{label_num}, 总数为{len(train_AL_3)}')
                    if label_num == 0:
                        pass
                    elif label_num == 1:  # 预测该子树样本与其根节点标签相同
                        y_test = np.delete(y[train_AL_3], 0)  # 测试集标签
                        y_predict = np.zeros(len(y_test), dtype=int) + y[train_AL_3[0]]
                        arr = y_test - y_predict
                        count = Counter(arr)[0]
                        print(f'第三层case1:子树{i}-{j}-{n}的准确率为{count / len(y_test)}, {count}/{len(y_test)}')
                        # y_predict_all[train_AL_2[lt3.AL[n]]] = y[train_AL_3[0]]
                        # SelectedInds_all = np.append(SelectedInds_all, train_AL_2[lt3.AL[n][0]])
                        y_predict_all[train_AL_3] = y[train_AL_3[0]]
                        SelectedInds_all = np.append(SelectedInds_all, train_AL_3[0])
                    else:
                        LT = lt.LeadingTree(X_train=X[train_AL_3], dc=dc_3[order3], lt_num=lt_num_3[order3])
                        LT.fit()
                        LTgammaPara = LT.density * LT.delta
                        selectedInds = DeLaLA_select(LTgammaPara, LT.density, LT.layer, y[train_AL_3], 2, label_num * 2,
                                                     0.5)  # selectedInds是相对索引
                        selectedInds_universe = train_AL_3[selectedInds]
                        X_train = X[selectedInds_universe]
                        y_train = y[selectedInds_universe]
                        lmca = lm.LMCA(dimension=2, init_method="kpca", verbose=True, max_iter=100, stepsize=1.E-2,
                                       nn_active=False, length_scale=length_scale_3[order3], k=1)
                        lmca.fit(X_train, y_train)
                        order3 += 1
                        X_test = np.delete(X[train_AL_3], selectedInds, axis=0)  # 去除训练集后得到测试集样本
                        y_test = np.delete(y[train_AL_3], selectedInds, axis=0)  # 测试集标签
                        y_predict = np.zeros(len(y_test), dtype=int) - 1  # 预测标签

                        MatDist = EuclidianDistsq(X_train, X_test)  # 训练集和测试集对应的核矩阵
                        test_bnd_K = np.exp(-1 * lmca.length_scale * MatDist).T
                        B = test_bnd_K.dot(lmca.Omega)  # 降维后的测试集
                        A = lmca.K.dot(lmca.Omega)  # 降维后的训练集

                        # 为每个测试样本找到欧氏距离最近的训练样本，预测二者标签相同。
                        D = EuclidianDist2(B, A)
                        Pa = np.zeros(len(y_predict), dtype=int)  # Pa[i]代表距离测试样本i最近的训练样本的索引
                        for m in range(len(y_predict)):
                            index1 = np.argmin(D[m])
                            Pa[m] = index1

                        y_predict = y_train[Pa]
                        arr = y_test - y_predict
                        count = Counter(arr)[0]
                        print(
                            f'第三层case2:子树{i}-{j}-{n}类别数为{len(np.unique(y[train_AL_3]))}, 准确率为{count / len(y_test)}, {count}/{len(y_test)}'
                            f'*****dc={LT.dc}, lt_num={LT.lt_num}, length_scale={lmca.length_scale}')
                        index_predict = np.delete(train_AL_3, selectedInds, axis=0)
                        y_predict_all[index_predict] = y_predict
                        SelectedInds_all = np.append(SelectedInds_all, selectedInds_universe)

index_None = np.zeros(0, dtype=int)
for i in range(len(remove_index_all)):
    if remove_index_all[i] == -1:
        index_None = np.append(index_None, i)

remove_index_all = np.delete(remove_index_all, index_None)
y_remove_Select = np.delete(y_predict_all, np.append(SelectedInds_all, remove_index_all))
D = EuclidianDist2(X[remove_index_all], np.delete(X, np.append(SelectedInds_all, remove_index_all), axis=0))
Pa = np.zeros(len(remove_index_all), dtype=int)
for i in range(len(remove_index_all)):
    index1 = np.argmin(D[i])
    Pa[i] = index1

y_predict_all[remove_index_all] = y_remove_Select[Pa]
arr1 = y[remove_index_all] - y_predict_all[remove_index_all]
count0 = Counter(arr1)[0]
print(f'单独预测的准确率为{count0 / len(arr1)}, {count0}/{len(arr1)}')

# 总准确率
y_predict_all = np.delete(y_predict_all, SelectedInds_all)
y_test_all = np.delete(y, SelectedInds_all)
arr = y_test_all - y_predict_all
count = Counter(arr)[0]
print(f'共选择{len(SelectedInds_all)}个点, 准确率为{count / len(y_test_all)}, {count}/{len(y_test_all)}')

t3 = time.time()
print('计算完毕,耗时', t3 - t1)
