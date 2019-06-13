#!/usr/bin/python3
import os
import numpy as np
import pickle

import csv
from predictor_new import DatasetStru



def create_traj(X, Y, Z, max_min = None, mean_std = None):
    xs = []
    x_lens = 0
    x_mean = np.zeros(3)
    x_var = np.zeros(3)
    x_ratio = 0.0
    if mean_std is not None:
        x_mean = mean_std[:, 0]
        x_var = mean_std[:, 1]

    if len(X) == len(Y) and len(X) == len(Z):
        for (x,y,z) in zip(X,Y,Z):
            pos = np.asarray([x, y, z])

            if max_min is not None and mean_std is None:
                pos = (pos-max_min[:,1])/(max_min[:,0]-max_min[:,1])
            elif mean_std is not None and max_min is None:
                pos = (pos - mean_std[:, 0]) / mean_std[:, 1]

            xs.append(pos)
        x_lens = len(xs)

        return DatasetStru(xs, x_lens, x_mean, x_var, x_ratio)
    else:
        print("error data length")
        return 0

def find_max_min(data, last_max_min):
    new_max_min = np.copy(last_max_min)
    data_max = max(data)
    data_min = min(data)

    if data_max>last_max_min[0]:
        new_max_min[0] = data_max
    if data_min<last_max_min[1]:
        new_max_min[1] = data_min
    return new_max_min

def find_max_min(delta_x, delta_y, delta_z):
    max_min = np.zeros(3,2)#xyz;max,min
    max_min[0][0] = max(delta_x)
    max_min[1][0] = max(delta_y)
    max_min[2][0] = max(delta_z)
    max_min[0][1] = min(delta_x)
    max_min[1][1] = min(delta_y)
    max_min[2][1] = min(delta_z)
    return max_min


def find_mean_std(delta_x, delta_y, delta_z):
    mean_std = np.zeros((3,2)) #xyz;mean;std

    mean_std[0][0] = np.mean(np.asarray(delta_x))
    mean_std[1][0] = np.mean(np.asarray(delta_y))
    mean_std[2][0] = np.mean(np.asarray(delta_z))
    mean_std[0][1] = np.std(np.asarray(delta_x))
    mean_std[1][1] = np.std(np.asarray(delta_y))
    mean_std[2][1] = np.std(np.asarray(delta_z))

    return mean_std

def get_delta(data):
    return (data[1:]-data[0]).tolist()


def load_file(f_csv):
    '''
    :param f_csv:
    :return:
    '''
    delta_x = []
    delta_y = []
    delta_z = []
    for r in f_csv:
        if "x:" in r[0]:
            x = np.asarray(str2float(r[1:]))
            delta_x.extend(get_delta(x))
        elif "y:" in r[0]:
            y = np.asarray(str2float(r[1:]))
            delta_y.extend(get_delta(y))
        elif "z:" in r[0]:
            z = np.asarray(str2float(r[1:]))
            delta_z.extend(get_delta(z))

    return delta_x, delta_y, delta_z


def create_set(f_csv, max_min = None, mean_std = None):
    dataset = []
    delta_x, delta_y, delta_z = [], [], []
    for r in f_csv:
        if "x:" in r[0]:
            x = np.asarray(str2float(r[1:]))
            delta_x = get_delta(x)
        elif "y:" in r[0]:
            y = np.asarray(str2float(r[1:]))
            delta_y = get_delta(y)
        elif "z:" in r[0]:
            z = np.asarray(str2float(r[1:]))
            delta_z = get_delta(z)
        elif "next" in r[0]:
            dataset.append(create_traj(delta_x, delta_y, delta_z, max_min, mean_std))
            delta_x, delta_y, delta_z = [], [], []
    return dataset


def str2float(list):
    float_list = [float(i) for i in list]
    return float_list

if __name__ == '__main__':
    csv_filename = "/home/xuan/Documents/traj_data/trajs_4.csv"

    # with open(csv_filename, 'r') as f:
    #     f_csv = csv.reader(f, delimiter=',')
    #     max_min = load_file(f_csv)

    with open(csv_filename, 'r') as f:
        f_csv = csv.reader(f, delimiter=',')
        delta_x, delta_y, delta_z = load_file(f_csv)
        mean_std = find_mean_std(delta_x, delta_y, delta_z)

    with open(csv_filename, 'r') as f:
        f_csv2 = csv.reader(f, delimiter=',')
        dataset = create_set(f_csv2, max_min = None, mean_std = mean_std)


    #todo: running std

    print("save dataset...")
    pickle.dump(dataset,
                open("./pred/" + "/dataset_mp" + ".pkl", "wb"))









