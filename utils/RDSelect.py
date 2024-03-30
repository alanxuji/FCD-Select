import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import pairwise_distances_argmin

def find_all_occurrences(arr, target):
    return [i for i, val in enumerate(arr) if val == target]


def FindDivergent(X,labels, selectedInds):
    K = len(np.unique(labels))
    ndim = np.shape(X)[1]
    MaxSize = 0
    centroid = np.zeros(ndim)
    ClusterInds2Find=[]
    for c in  range(K):
        ClusterInds = find_all_occurrences(labels, c)
        if set(ClusterInds).isdisjoint(set(selectedInds)):
            CurSize = len(ClusterInds)
            if CurSize>MaxSize:
                MaxSize=CurSize
                GreatestClust = X[ClusterInds]
                centroid = np.sum(GreatestClust, axis=0)/MaxSize
                ClusterInds2Find = ClusterInds
    InterInd = FindNearesetX2Centroids(X[ClusterInds2Find], centroid.reshape((1,4)))
    InterInd = InterInd[0]
    return ClusterInds2Find[InterInd]

def FindNearesetX2Centroids(X, centroids):
    K = np.shape(centroids)[0]
    N = np.shape(X)[0]
    NearInds = np.zeros(K, dtype=int) -1
    DistVec = np.zeros(N, dtype=float) -1
    for i in range(K):
        for j in range(N):
            DistVec[j] = np.linalg.norm(X[j] - centroids[i,])
        NearInds[i] = np.argmin(DistVec)
    return NearInds


def RDSeclet(X,d,M):
    selectedInds = np.zeros(M,dtype=int) -1
    kmeans = KMeans(n_clusters=d, random_state=0)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    selectedInds[0:d] = FindNearesetX2Centroids(X, centroids)

    #print(kmeans.labels_)

    for nClust in range(d,M,1):
        kmeans = KMeans(n_clusters=nClust+1, random_state=0)
        kmeans.fit(X)
        print(kmeans.cluster_centers_)
        print()
        curSelInds = selectedInds[:nClust]
        selectedInds[nClust] = FindDivergent(X, kmeans.labels_,curSelInds)
    return selectedInds

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(RDSeclet(X,4,8))