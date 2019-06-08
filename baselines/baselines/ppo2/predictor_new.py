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





class Predictor(object):
    def __init__(self, batch_size, out_max_timestep, train_flag,
                 reset_flag=False, epoch=20, iter_start=0,
                 lr=0.001, load=False):
        ## extract FLAGS
        if iter_start == 0 and lr<0.001:
            self.start_iter = 20
        else:
            self.start_iter = iter_start * epoch
        self.iteration = 0
        self.dataset_length = 0

        self.batch_size = batch_size
        self.in_timesteps_max = 300
        self.out_timesteps = out_max_timestep
        self.train_flag = train_flag
        self.epochs = epoch
        self.lr = lr
        self.validate_ratio = 0.2

        self.num_units=64

        self.in_dim=10
        self.out_dim=10

        self.train_model = KP.TrainRNN(self.batch_size,
                                       self.in_dim, self.out_dim, self.num_units, num_layers=1, load=load)

        self.inference_model = KP.PredictRNN(self.batch_size,
                                       self.in_dim, self.out_dim, self.num_units, num_layers=1)


    def run_training(self):
        ## check whether in training
        if not self.train_flag:
            print("Not in training process,return...")
            return 0

        ## load dataset
        self._load_train_set()
        print("trajectory numbers: ", len(self.dataset))
        valid_len = int(self.validate_ratio * len(self.dataset))

        train_set = self._process_dataset(self.dataset[0:-valid_len])
        valid_set = self._process_dataset(self.dataset[-valid_len:-1])

        self.train_model.training(X=train_set[0], Y=train_set[1], epochs=5)


    def run_prediction(self):
        ## load dataset
        self._load_train_set()
        print("trajectory numbers: ", len(self.dataset))
        valid_len = int(self.validate_ratio * len(self.dataset))

        valid_set = self._process_dataset(self.dataset[-valid_len:-1])

        self.inference_model.predict(X=valid_set[0], Y=valid_set[0])

    def _process_dataset(self, trajs):
        xs, ys, x_lens, xs_start = [], [], [], []
        for traj in trajs:
            x, y, x_len, x_start = self._feed_one_data(traj)
            xs.append(x)
            ys.append(y)
            x_lens.append(x_len)
            xs_start.append(x_start)

        xs=np.asarray(xs, dtype=np.float32)
        xs.reshape((len(trajs),self.in_timesteps_max, self.in_dim))
        ys=np.asarray(ys)
        ys.reshape((len(trajs), self.in_timesteps_max, self.out_dim))
        x_lens=np.asarray(x_lens, dtype=np.float32)
        xs_start=np.asarray(xs_start)
        return [xs, ys, x_lens, xs_start]

    def _feed_one_data(self, data):
        # pading dataset
        length = data.x_len
        if length > self.in_timesteps_max:
            x_seq = data.x[0:self.in_timesteps_max,:]
        else:
            x_seq = data.x
        x_start = x_seq[0,:]

        x = x_seq[1:-1,:] - x_start
        y = x_seq[2:,:] - x_start

        x = self._padding(x, self.in_timesteps_max, 0.0)
        y = self._padding(y, self.in_timesteps_max, 0.0)

        return x, y,length, x_start

    def _padding(self, seq, new_length, my_value=None):
        old_length = len(seq)
        value = seq[-1, :]
        if not my_value is None:
            value.fill(my_value)
        value = np.expand_dims(value, axis=0)

        for _ in range(old_length, new_length):
                seq = np.append(seq, value, axis=0)
        return seq

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
                              iter_start=args.iter, lr=args.lr, load=args.load)

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
        rnn_model = Predictor(1024, out_steps, train_flag=True, reset_flag=False, epoch=args.epoch,
                              iter_start=args.iter, lr=args.lr, load=args.load)

        rnn_model.run_prediction()
        # plot and check dataset
        # rnn_model.plot_dataset()

        # rnn_model.init_sess()
        # rnn_model.load()
        # rnn_model.run_test()
        #
