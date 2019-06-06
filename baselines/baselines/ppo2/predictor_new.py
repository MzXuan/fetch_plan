# -*- coding: utf-8 -*-
import time, os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import joblib
import pickle
import os
import time
import random
import numpy as np

import tensorflow as tf

import visualize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

import keras_predictor as KP

class DatasetStru(object):
    def __init__(self, x, x_len, x_mean, x_var, x_ratio):
        """
        :param x: shape = (self.in_timesteps_max, self.in_dim)
        :param x_len: shape = 1
        :param x_mean: shape = (self.in_dim)
        :param x_var:  shape = (self.in_dim)
        """
        self.x = np.asarray(x)
        self.x_len = x_len
        self.x_mean = x_mean
        self.x_var = x_var
        self.x_ratio = x_ratio


    def padding(self, seq, new_length):
        old_length = len(seq)
        value = seq[-1, :]
        value = np.expand_dims(value, axis=0)
        for _ in range(old_length, new_length):
            seq = np.append(seq, value, axis=0)

        return seq


class Predictor(object):
    def __init__(self, batch_size, out_max_timestep, train_flag,
                 reset_flag=False, epoch=20, iter_start=0,
                 lr=0.001):
        ## extract FLAGS
        if iter_start == 0 and lr<0.001:
            self.start_iter = 20
        else:
            self.start_iter = iter_start * epoch
        self.iteration = 0
        self.dataset_length = 0

        self.batch_size = batch_size
        # self.in_timesteps_max = max_timestep
        self.out_timesteps = out_max_timestep
        self.train_flag = train_flag
        self.epochs = epoch
        self.lr = lr
        self.validate_ratio = 0.2

        self.num_units=64

        self.in_dim=3
        self.out_dim=3

        self.train_model = KP.TrainRNN(self.batch_size,
                                       self.in_dim, self.out_dim, self.num_units, num_layers=1)

    def run_training(self):
        ## check whether in training
        if not self.train_flag:
            print("Not in training process,return...")
            return 0

        ## load dataset
        self._load_train_set()


    def load_dataset(self, file_name):
        ## load dataset
        try:
            dataset = pickle.load(open(os.path.join("./pred/", file_name), "rb"))
            return dataset
        except:
            print("Can not load dataset. Please first run the training stage to save dataset.")

    def _load_train_set(self):
        ## check saved data set
        filelist = [f for f in os.listdir("./pred/") if f.endswith(".pkl")]
        num_sets = len(filelist)

        self.dataset = []
        for idx in range(num_sets):
            dataset = self.load_dataset(filelist[idx])
            if dataset == 0:
                return 0
            else:
                self.dataset.extend(dataset)

    def plot_dataset(self):
        self._load_train_set()
        #plot dataset
        for idx, data in enumerate(self.dataset):
            if idx%10 == 0:
                visualize.plot_3d_eef(data.x)
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--iter', default=0, type=int)
    parser.add_argument('--model_name', default='test1', type=str)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    test_flag=args.test
    out_steps=100

    if not os.path.isdir("./pred"):
        os.mkdir("./pred")

    if not test_flag:
        # create and initialize session
        rnn_model = Predictor(1024, out_steps, train_flag=True, reset_flag=False, epoch=args.epoch,
                              iter_start=args.iter, lr=args.lr)

        # rnn_model.init_sess()

        # #-----------------for debug--------------
        # rnn_model.plot_dataset()
        #
        # #-----end debug------------------------

        # if args.load:
        #     try:
        #         rnn_model.load()
        #         print("load model successfully")
        #     except:
        #         rnn_model.init_sess()

        rnn_model.run_training()

    else:
        print("start testing...")
        # plot all the validate data step by step
        rnn_model = Predictor(1, out_steps, train_flag=False, reset_flag=False, epoch=args.epoch)

        # plot and check dataset
        # rnn_model.plot_dataset()

        # rnn_model.init_sess()
        # rnn_model.load()
        # rnn_model.run_test()
        #
