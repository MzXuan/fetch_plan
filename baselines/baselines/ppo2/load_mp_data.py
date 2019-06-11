#!/usr/bin/python3
import os
import numpy as np
import pickle

import csv
from predictor_new import DatasetStru



def create_traj(X, Y, Z, max_min):
    # print("data list")
    # print(data_list)
    xs = []
    x_lens = 0
    x_mean = np.zeros(3)
    x_var = np.zeros(3)
    x_ratio = 0.0

    if len(X) == len(Y) and len(X) == len(Z):
        for (x,y,z) in zip(X,Y,Z):
            pos = np.asarray([x, y, z])
            pos = (pos-max_min[:,0])/(max_min[:,0]-max_min[:,1])
            xs.append(pos)
        x_lens = len(xs)
        # print("xs")
        # print(xs)
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

def load_file(f_csv):
    '''

    :param f_csv:
    :return:
    '''
    max_min = np.zeros((3, 2))  # x,y,z;max;min
    x, y, z = [], [], []
    for r in f_csv:
        if "x:" in r[0]:
            x = str2float(r[1:])
        elif "y:" in r[0]:
            y = str2float(r[1:])
        elif "z:" in r[0]:
            z = str2float(r[1:])
        elif "next" in r[0]:
            max_min[0, :] = find_max_min(x, max_min[0, :])
            max_min[1, :] = find_max_min(y, max_min[1, :])
            max_min[2, :] = find_max_min(z, max_min[2, :])
            x, y, z = [], [], []
    print("max_min")
    print(max_min)
    return max_min

def create_normalize_set(f_csv, max_min):
    dataset = []
    x, y, z = [], [], []
    for r in f_csv:
        if "x:" in r[0]:
            x = str2float(r[1:])
        elif "y:" in r[0]:
            y = str2float(r[1:])
        elif "z:" in r[0]:
            z = str2float(r[1:])
        elif "next" in r[0]:
            dataset.append(create_traj(x, y, z, max_min))
            x, y, z = [], [], []

    return dataset




def str2float(list):
    float_list = []
    float_list = [float(i) for i in list]
    return float_list

if __name__ == '__main__':
    csv_filename = "/home/xuan/Documents/traj_data/trajs_4.csv"

    with open(csv_filename, 'r') as f:
        f_csv = csv.reader(f, delimiter=',')
        max_min = load_file(f_csv)

    with open(csv_filename, 'r') as f:
        f_csv2 = csv.reader(f, delimiter=',')
        dataset = create_normalize_set(f_csv2, max_min)


    #todo: running std

    print("save dataset...")
    pickle.dump(dataset,
                open("./pred/" + "/dataset_mp" + ".pkl", "wb"))









