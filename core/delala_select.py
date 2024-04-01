import numpy as np


def DeLaLA_select(gamma, rho, layer, Y, k, l, rootWeight):
    """
    select the samples to label according to Objective Function
    :param gamma: center potential
    :param rho: local density
    :param layer: layer index of each sample
    :param alhpa: parameter of the divergence item
    :param Y: labels
    :param k: k for LMCA
    :param l: given number of samples to be labeled
    :return: labeled samples Indics
    """
    N = len(Y)
    psi = np.zeros(N, dtype=float)
    np.divide(rho, layer, psi)
    C = len(np.unique(Y))
    label_unique = np.unique(Y)

    y = [x for x in Y]

    for i in range(len(y)):
        for j in range(C):
            if y[i] == label_unique[j]:
                y[i] = j
    LabeledPerClass = np.zeros((C, k), dtype=int) - 1
    p = l - C * k  ## number of global selection
    globalSelected = np.zeros(p, dtype=int) - 1
    gamma = (gamma - min(gamma)) / (max(gamma) - min(gamma))
    BigValue = 10
    hgamma = np.zeros(N, dtype=float)
    for i in range(N):
        if np.abs(gamma[i] - 1) < 1E-3:
            hgamma[i] = BigValue
        else:
            if np.abs(gamma[i]) < 1E-3:
                hgamma[i] = 0
            else:
                hgamma[i] = 1.0 / np.log(gamma[i])

    sortedInds = SortSmallXOR(hgamma, psi, rootWeight)
    Cursor = 0
    # hgammaCursor =0
    labeled = 0
    while labeled < l:
        classID = y[sortedInds[Cursor]]
        if LabeledPerClass[classID, k - 1] > -1:  ### selection has done for the class.
            if p > 0:
                if globalSelected[p - 1] > -1:  ### global selection done
                    Cursor += 1
                    continue
                else:
                    for i in range(p):
                        if globalSelected[i] == -1:
                            globalSelected[i] = sortedInds[Cursor]  ###selected one sample in global selection
                            labeled += 1
                            Cursor += 1
                            break
            else:
                Cursor += 1
        else:
            for i in range(k):
                if LabeledPerClass[classID, i] == -1:
                    LabeledPerClass[classID, i] = sortedInds[Cursor]  ###selected one sample for a given class
                    labeled += 1
                    Cursor += 1
                    break
    result = np.append(LabeledPerClass.flatten(), globalSelected)
    return result

def Fuzzy_select_old(gamma, rho, layer, Y, k, l, rootWeight):
    """
    select the samples to label according to Objective Function
    :param gamma: center potential
    :param rho: local density
    :param layer: layer index of each sample
    :param alhpa: parameter of the divergence item
    :param Y: labels
    :param k: k for LMCA
    :param l: given number of samples to be labeled
    :return: labeled samples Indics
    """
    N = len(Y)
    psi = np.zeros(N, dtype=float)
    SMALLValue = 1e-8
    ###revised, reversion
    for i in range(N):
        if rho[i] <SMALLValue:
            rho[i] = SMALLValue
    np.divide(layer, rho,  psi)
    psi = np.log(psi)
    ### added log

    C = len(np.unique(Y))
    label_unique = np.unique(Y)

    y = [x for x in Y]

    for i in range(len(y)):
        for j in range(C):
            if y[i] == label_unique[j]:
                y[i] = j
    LabeledPerClass = np.zeros((C, k), dtype=int) - 1
    p = l - C * k  ## number of global selection
    globalSelected = np.zeros(p, dtype=int) - 1
    #gamma = (gamma - min(gamma)) / (max(gamma) - min(gamma))

    hgamma = np.zeros(N, dtype=float)
    for i in range(N):
        if gamma[i] < 1E-5:
            gamma[i] = SMALLValue
    #     else:
    #         if np.abs(gamma[i]) < 1E-3:
    #             hgamma[i] = 0
    #         else:
    #             #hgamma[i] = 1.0 / np.log(gamma[i])
    #             hgamma[i] = np.log(gamma[i])
    hgamma = np.log(gamma)
    #hgamma = gamma
    sortedInds = SortSmallXOR2(hgamma, psi, rootWeight)
    Cursor = 0
    # hgammaCursor =0
    labeled = 0
    while labeled < l:
        classID = y[sortedInds[Cursor]]
        if LabeledPerClass[classID, k - 1] > -1:  ### selection has done for the class.
            if p > 0:
                if globalSelected[p - 1] > -1:  ### global selection done
                    Cursor += 1
                    continue
                else:
                    for i in range(p):
                        if globalSelected[i] == -1:
                            globalSelected[i] = sortedInds[Cursor]  ###selected one sample in global selection
                            labeled += 1
                            Cursor += 1
                            break
            else:
                Cursor += 1
        else:
            for i in range(k):
                if LabeledPerClass[classID, i] == -1:
                    LabeledPerClass[classID, i] = sortedInds[Cursor]  ###selected one sample for a given class
                    labeled += 1
                    Cursor += 1
                    break
    result = np.append(LabeledPerClass.flatten(), globalSelected)
    return result

def SortSmallXOR2(a, b, rootWeight=0.5):
    '''
    a and b are exclusively small. The function sort the values of a XOR b in the sense of being small
    :param a:
    :param b:
    :param rootWeight: if >0, tends to include more roots than divergent samples
    :return: the indics of an array, first elements are small in a or small in b.
    '''


    normalA = (a - min(a)) / (max(a) - min(a))
    normalB = (b - min(b)) / (max(b) - min(b))

    ratio = np.mean(normalA) / np.mean(normalB)

    print ("A to B ratio: ", np.mean(normalA)/np.mean(normalB))

    Threshold = 2
    if ratio>Threshold or ratio<1/Threshold:
        normalB = normalB*ratio

    y = rootWeight * normalA * (1 - normalB) + (1 - rootWeight) * normalB * (1 - normalA)
    inds = np.argsort(y)
    # print('Inds:',inds)
    # ysort = np.sort(y)
    # print('ysort:',ysort)
    # result = inds[::-1] ###revised
    return inds
def SortSmallXOR(a, b, rootWeight=0.5):
    '''
    a and b are exclusively small. The function sort the values of a XOR b in the sense of being small
    :param a:
    :param b:
    :param rootWeight: if >0, tends to include more roots than divergent samples
    :return: the indics of an array, first elements are small in a or small in b.
    '''

    ###select via XOR. min-max normalization failed
    normalA = (a - np.mean(a)) / np.std(a)
    normalB = (b - np.mean(b)) / np.std(b)

    if np.mean(normalB)!=0:
        print ("A to B ratio: ", np.mean(normalA)/np.mean(normalB))

    y = rootWeight * normalA * (1 - normalB) + (1 - rootWeight) * normalB * (1 - normalA)
    inds = np.argsort(y)
    # print('Inds:',inds)
    # ysort = np.sort(y)
    # print('ysort:',ysort)
    result = inds[::-1] ###revised
    return result

def find_all_occurrences(arr, target):
    return [i for i, val in enumerate(arr) if val == target]

def Fuzzy_select(gamma, rho, layer, Y, k, l, rootWeight):
    """
    select the samples to label according to Objective Function
    :param gamma: center potential
    :param rho: local density
    :param layer: layer index of each sample
    :param alhpa: parameter of the divergence item
    :param Y: labels
    :param k: k for LMCA
    :param l: given number of samples to be labeled
    :return: labeled samples Indics
    """
    N = len(Y)
    psi = np.zeros(N, dtype=float)
    SMALLValue = 1e-8
    ###revised, reversion
    for i in range(N):
        if rho[i] <SMALLValue:
            rho[i] = SMALLValue
    np.divide(layer, rho,  psi)
    psi = np.log(psi)
    ### added log

    C = len(np.unique(Y))
    label_unique = np.unique(Y)

    y = [x for x in Y]

    for i in range(len(y)):
        for j in range(C):
            if y[i] == label_unique[j]:
                y[i] = j
    LabeledPerClass = np.zeros((C, k), dtype=int) - 1
    p = l - C * k  ## number of global selection
    globalSelected = np.zeros(p, dtype=int) - 1

    for i in range(N):
        if gamma[i] < 1E-5:
            gamma[i] = SMALLValue

    hgamma = np.log(gamma)
    sortedInds = SortSmallXOR2(hgamma, psi, rootWeight)
    sortedY = Y[sortedInds]
    for cl in y:

        clInds = find_all_occurrences(sortedY, cl)
        LabeledPerClass[cl,:] = sortedInds[clInds[:k]] ###this is important!!
    ClassSamples =  LabeledPerClass.flatten()
    NewSortedInds = [elment for elment in sortedInds if elment not in ClassSamples]
    if p>0:
        globalSelected = NewSortedInds[:p]

    result = np.append(ClassSamples, globalSelected)
    return result