#### Modified AlanXu@SKL.PBD 2022-9-12

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
from scipy.linalg import fractional_matrix_power
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import scipy.io as scio
import datetime
import time
from sklearn import datasets


alpha = 0.5
sigma = 0.05
lpc = 2
Runtimes = 50

def InitInputYDeterm(Y,givenInds):
    SampleSize = len(Y)
    classNum = len(np.unique(Y))
    Y_input = np.zeros((SampleSize, classNum), dtype=int)
    for ind in givenInds:
        classID = Y[ind]
        Y_input[ind, classID] = 1
    return Y_input


def InitInputY(classNum,Y, lpc):
    """

    :param classNum: class Number
    :param Y: real labels vector, starting from 0
    :param lpc: num of labeled per class
    :return: randomly selected labeling Matrix
    """
    SampleSize = len(Y)
    Y_input = np.zeros((SampleSize,classNum),dtype=int)
    for classID in range(classNum):
        classInds = [i for i in range(SampleSize) if Y[i]==classID]
        np.random.shuffle(classInds)
        selectedCur = classInds[:lpc]
        Y_input[selectedCur, classID] = 1
    return Y_input

#X = np.loadtxt("moon_data.txt")
#Y = np.loadtxt("class.txt")

# X, Y = make_moons(n, shuffle=True, noise=0.1, random_state=None)


wine = datasets.load_wine()
X = wine.data
Y = wine.target

# data = scio.loadmat("D:\\GZU2022b\\LMLF\\venv\\data\\ORL_32x32.mat")
# X, Y = np.array(data["fea"]), np.array(data["gnd"])
n = X.shape[0]
classNum = len(np.unique(Y))
# region show plot
# color = ['red' if l == 0 else 'blue' for l in Y]
# plt.scatter(X[0:,0], X[0:,1], color=color)
# #plt.savefig("ideal_classification.pdf", format='pdf')
# plt.show()
# endregion

## select X and Y
#Y_input = np.concatenate(((Y[:n_labeled,None] == np.arange(2)).astype(float), np.zeros((n-n_labeled,2))))


t1 = time.time()
for runtime in range(Runtimes):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    dm = cdist(X, X, 'euclidean')

    rbf = lambda x, sigma: math.exp((-x)/(2*(math.pow(sigma,2))))
    vfunc = np.vectorize(rbf)
    W = vfunc(dm, sigma)
    np.fill_diagonal(W, 0)

    sum_lines = np.sum(W,axis=1)
    D = np.diag(sum_lines)

    D = fractional_matrix_power(D, -0.5)
    S = np.dot(np.dot(D,W), D)


    ACCArray = np.zeros((Runtimes,1),dtype=float)
    #givenInds = np.array([ 113, 202, 465, 466, 480, 492, 513, 510, 577, 568, 716, 673, 966, 790, 1153, 1382, 1444, 1440, 1463,1483 ])
    givenInds = np.array([ 12,  57, 125,  91, 148, 140]) ###Find in DeLaLA-Wine.py

for runtime in range(Runtimes):
    #Y_input = InitInputY(classNum, Y, lpc)

    Y_input = InitInputYDeterm(Y, givenInds)
    n_iter = 400

    F = np.dot(S, Y_input)*alpha + (1-alpha)*Y_input
    for t in range(n_iter):
        F = np.dot(S, F)*alpha + (1-alpha)*Y_input


    Y_result = np.zeros_like(F)

    Y_v = [np.argmax(F [i,:]) for i in range(n)]
    ##############################################
    t2 = time.time()
    print("Time elapse: ", t2 - t1)
    ############################################################
    arr = Y - Y_v
    count = Counter(arr)[0]
    labeledNum = classNum*lpc
    print("准确率为", (count-labeledNum) / (n-labeledNum),", ",count-labeledNum,"/", n-labeledNum)
    ACCArray[runtime] = (count-labeledNum) / (n-labeledNum)

print("总的准确率为", np.mean(ACCArray),", std ", np.var(ACCArray))


