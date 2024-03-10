from csv import reader

import numpy as np


def load_csv(file_path):
    # LOAD Data
    dataset = list()
    with open(file_path, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return np.array(dataset)


def load_data(file_path, label_index=None, map_label=True):
    dataset = load_csv(file_path)
    if label_index is None:
        label_index = dataset.shape[1] - 1
    y = dataset[:, label_index].astype(int)
    X = np.delete(dataset, label_index, axis=1)

    if map_label:
        ###mapping the labels
        Unilabels = np.unique(y)
        for i in range(len(y)):
            y[i] = int(list(Unilabels).index(y[i]))
        y = np.array(y).astype(int)
    return X, y


def check_numpy(*data):
    new_data = []
    for item in data:
        if item is None \
                or type(item) is np.ndarray \
                or type(item) is np.matrix:
            new_data.append(item)
        else:
            new_data.append(np.array(item))

    if len(new_data) == 1:
        return new_data[0]
    return tuple(new_data)


def del_invalid(samples, invalid_cols=None):
    if invalid_cols is not None and len(invalid_cols) == 0:
        return samples
    elif invalid_cols is not None and len(invalid_cols) > 0:
        return np.delete(samples, invalid_cols, axis=1)
    invalid_cols = []
    for i in range(samples.shape[1]):
        uni = np.unique(samples[:, i])
        if uni.shape[0] == 1:
            invalid_cols.append(i)
    return np.delete(samples, invalid_cols, axis=1), invalid_cols


def standard_norm(samples, mean=None, std=None):
    return_flag = False
    samples = check_numpy(samples).astype(np.float32)
    if mean is None:
        return_flag = True
        mean = np.mean(samples, axis=0)
    if std is None:
        return_flag = True
        std = np.std(samples, axis=0)
    if return_flag:
        return _standard_norm(samples, mean, std), mean, std
    return _standard_norm(samples, mean, std)


def _standard_norm(samples, mean, std):
    samples = (samples - mean) / std
    return samples


def max_min_norm(samples, data_max=None, data_min=None):
    return_flag = False
    samples = check_numpy(samples).astype(np.float32)
    if data_max is None:
        return_flag = True
        data_max = np.max(samples, axis=0)
    if data_min is None:
        return_flag = True
        data_min = np.min(samples, axis=0)
    if return_flag:
        return _max_min_norm(samples, data_max, data_min), data_max, data_min
    return _max_min_norm(samples, data_max, data_min)


def _max_min_norm(samples, data_max, data_min):
    samples = (samples - data_min) / (data_max - data_min)
    return samples


def euclidian_dist(X1, X2):
    euclidianDistSquare = euclidian_dist_square(X1, X2)
    return np.sqrt(euclidianDistSquare)


def euclidian_dist_square(X1, X2):
    ###Using broadcasting, simpler and faster!
    tempM = np.sum(X1 ** 2, 1, dtype='float32').reshape(-1, 1)  ##The number of rows is unknown, only the number of columns is 1
    tempN = np.sum(X2 ** 2, 1, dtype='float32')  # X2 ** 2: element-wise square, sum(_,1): Adds in row direction, but ends up with row vectors
    sqdist = tempM + tempN - 2 * np.dot(X1, X2.T).astype('float32')
    sqdist[sqdist < 0] = 0
    return sqdist


def euclidian_dist_squareT(X1, X2):
    euclidianDistSquare = euclidian_dist_square(X1, X2)
    return euclidianDistSquare.T
