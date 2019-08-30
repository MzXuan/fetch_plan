import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import os
import time
import csv

import pred_flags

import numpy as np
np.random.seed(pred_flags.random_seed)


import tensorflow as tf
tf.set_random_seed(pred_flags.random_seed)

import keras_seq2seq as KP
import keras_simpleRNN as KS

from pred_base import PredBase



NUM_UNITS = pred_flags.num_units
NUM_LAYERS = pred_flags.num_layers


class ShortPred(PredBase):
    def __init__(self, batch_size, in_max_timestep, out_timesteps, train_flag,
                 step=1, epoch=20, iter_start=0,
                 lr=0.001, load=False, model_name="short_pred"):
        in_max_timestep = 3
        super(ShortPred, self).__init__(batch_size, in_max_timestep, 1, train_flag,
                 step, epoch, iter_start,lr, load, model_name)

        train_model = KS.SimpleRNN(self.batch_size,
                                     self.in_dim, self.out_dim, initial_epoch=self.start_iter, load=self.load,
                                     model_name=self.model_name)

        inference_model = KS.SimpleRNN(self.batch_size,
                  self.in_dim, self.out_dim, initial_epoch=self.start_iter, load=self.load,
                  model_name=self.model_name)

        if self.load:
            train_model.load_model()
        inference_model.load_model()

        self.init_model(train_model, inference_model)

    def run_online_prediction(self, batched_seqs, visial_flags = False):
        '''
        :param batched_seqs: raw x (batch size * input shape)
        :param batched_goals: raw goals / true goal (batch size * goal shape)
        :return: reward of this prediction, batch size * 1
        '''
        rewards = []
        batched_seqs_normal = []
        ys = []
        for seq in batched_seqs:
            if seq.shape[0]>0:
                ys.append(seq[-1, -3:]) #y
            else:
                ys.append(0.0)
            seq_normal = (seq[:-1, -3:] - self.x_mean) / self.x_var #x

            if seq_normal.shape[0] < self.in_timesteps_max:
                seq_normal = self._padding(seq_normal, self.in_timesteps_max, 0.0)
            else:
                seq_normal = seq_normal[-self.in_timesteps_max:]
            batched_seqs_normal.append(seq_normal)

        batched_seqs_normal = np.asarray(batched_seqs_normal)
        ys_pred = self.inference_model.predict(X=batched_seqs_normal)

        # then we restore the origin x and calculate the reward
        for seq, y_pred, y_true in zip(batched_seqs, ys_pred, ys):
            if seq.shape[0] < 3:
                rewards.append(0.0)
            else:
                #calculated mean euclidean distance
                raw_y_pred = y_pred * self.x_var + self.x_mean
                rewards.append(np.linalg.norm(raw_y_pred-y_true))

        # print("rewards: ", rewards)
        # print("rewards.shape: ", len(rewards))


        rewards = np.asarray(rewards)
        return rewards

    def _feed_one_data(self, data, id_start = 0):
        # X: N step; Y: N+M step
        length = data.x_len
        x_seq = data.x
        x_full = data.x[:, -3:]

        step = self.step

        if length > self.in_timesteps_max:
            id_end = id_start + self.in_timesteps_max + step * self.out_timesteps
        else:
            id_end = length

        x_start = x_seq[0, -3:]

        x = x_seq[id_start:id_start + self.in_timesteps_max, -3:]
        y = x_seq[id_start + self.in_timesteps_max:id_end:step, -3:]

        x = self._padding(x, self.in_timesteps_max, 0.0)
        y = self._padding(y, self.out_timesteps)

        y = y.reshape(self.in_dim)

        return x, y, length, x_start, x_full




class LongPred(PredBase):
    def __init__(self, batch_size, in_max_timestep, out_timesteps, train_flag,
                 step=3, epoch=20, iter_start=0,
                 lr=0.001, load=False, model_name="seq2seq"):
        super(LongPred, self).__init__(batch_size, in_max_timestep, out_timesteps, train_flag,
                          step, epoch, iter_start, lr, load, model_name)

        train_model = KP.TrainRNN(self.batch_size,
                                       self.in_dim, self.out_dim, self.out_timesteps, self.num_units,
                                       initial_epoch=self.start_iter, num_layers=self.num_layers, load=self.load,
                                       model_name=self.model_name)

        inference_model = KP.PredictRNN(self.batch_size,
                                        self.in_dim, self.out_dim, self.in_timesteps_max, 2 * self.out_timesteps,
                                        self.num_units, num_layers=self.num_layers,
                                        model_name=model_name)
        if self.load:
            train_model.load_model()
        inference_model.load_model()

        self.init_model(train_model, inference_model)


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--epoch', default=20, type=int)
#     parser.add_argument('--lr', default=0.01, type=float)
#     parser.add_argument('--load', action='store_true')
#     parser.add_argument('--iter', default=0, type=int)
#     parser.add_argument('--model_name', default='test1', type=str)
#     parser.add_argument('--test', action='store_true')
#     args = parser.parse_args()
#
#     test_flag=args.test
#     out_steps=pred_flags.out_steps
#
#     if not os.path.isdir("./pred"):
#         os.mkdir("./pred")
#
#     # rnn_model.plot_dataset()
#
#     if not test_flag:
#
#         rnn_model = ShortPred(1024, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=out_steps,
#                               train_flag=True, epoch=args.epoch,
#                               iter_start=args.iter, lr=args.lr, load=args.load)
#
#         rnn_model.run_training()
#
#         print("start testing.....")
#         rnn_model2 = ShortPred(1, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=out_steps,
#                               train_flag=True, epoch=args.epoch,
#                               iter_start=args.iter, lr=args.lr, load=args.load)
#         rnn_model2.run_validation()
#
#     else:
#
#         rnn_model = ShortPred(1, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=out_steps,
#                               train_flag=True, epoch=args.epoch,
#                               iter_start=args.iter, lr=args.lr, load=args.load,)
#         print("start testing...")
#
#         rnn_model.run_validation()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--iter', default=0, type=int)
    parser.add_argument('--model_name', default='test1', type=str)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    test_flag=args.test
    out_steps=pred_flags.out_steps

    if not os.path.isdir("./pred"):
        os.mkdir("./pred")


    if not test_flag:

        rnn_model = LongPred(1024, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=out_steps,
                              train_flag=True, epoch=args.epoch,
                              iter_start=args.iter, lr=args.lr, load=args.load,
                              model_name=pred_flags.model_name)

        rnn_model.run_training()
        # rnn_model.plot_dataset()

        # print("start testing.....")
        # rnn_model2 = LongPred(1, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=out_steps,
        #                       train_flag=True, epoch=args.epoch,
        #                       iter_start=args.iter, lr=args.lr, load=args.load,
        #                       model_name=pred_flags.model_name)
        # rnn_model2.run_validation()


    else:

        rnn_model = LongPred(1, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=out_steps,
                              train_flag=True, epoch=args.epoch,
                              iter_start=args.iter, lr=args.lr, load=args.load,
                              model_name=pred_flags.model_name)
        print("start testing...")
        # plot all the validate data step by step

        # rnn_model.plot_dataset()
        rnn_model.run_validation()


