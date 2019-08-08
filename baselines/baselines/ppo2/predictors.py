import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import os
import time
import csv
import numpy as np

import keras_seq2seq as KP
import keras_simpleRNN as KS

from pred_base import PredBase

import pred_flags

NUM_UNITS = pred_flags.num_units
NUM_LAYERS = pred_flags.num_layers


class ShortPred(PredBase):
    def __init__(self, batch_size, in_max_timestep, out_timesteps, train_flag,
                 epoch=20, iter_start=0,
                 lr=0.001, load=False, model_name="test"):
        in_max_timestep = 3
        out_timesteps = 1
        super(ShortPred, self).__init__(batch_size, in_max_timestep, out_timesteps, train_flag,
                 epoch, iter_start,lr, load, model_name)

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



class LongPred(PredBase):
    def __init__(self, batch_size, in_max_timestep, out_timesteps, train_flag,
                 epoch=20, iter_start=0,
                 lr=0.001, load=False, model_name="test"):
        super(LongPred, self).__init__(batch_size, in_max_timestep, out_timesteps, train_flag,
                          epoch, iter_start, lr, load, model_name)

        train_model = KP.TrainRNN(self.batch_size,
                                       self.in_dim, self.out_dim, self.out_timesteps, self.num_units,
                                       initial_epoch=self.start_iter, num_layers=self.num_layers, load=self.load,
                                       model_name=self.model_name)

        inference_model = KP.PredictRNN(self.batch_size,
                                        self.in_dim, self.out_dim, self.in_timesteps_max, 3 * self.out_timesteps,
                                        self.num_units, num_layers=self.num_layers,
                                        model_name=model_name)
        if self.load:
            train_model.load_model()
        inference_model.load_model()

        self.init_model(train_model, inference_model)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epoch', default=10, type=int)
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

    # rnn_model.plot_dataset()

    if not test_flag:

        rnn_model = LongPred(1024, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=out_steps,
                              train_flag=True, epoch=args.epoch,
                              iter_start=args.iter, lr=args.lr, load=args.load,
                              model_name=pred_flags.model_name)

        rnn_model.run_training()

        print("start testing.....")
        rnn_model2 = LongPred(1, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=out_steps,
                              train_flag=True, epoch=args.epoch,
                              iter_start=args.iter, lr=args.lr, load=args.load,
                              model_name=pred_flags.model_name)
        rnn_model2.run_validation()


    else:

        rnn_model = LongPred(1, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=out_steps,
                              train_flag=True, epoch=args.epoch,
                              iter_start=args.iter, lr=args.lr, load=args.load,
                              model_name=pred_flags.model_name)
        print("start testing...")
        # plot all the validate data step by step

        # rnn_model.plot_dataset()
        rnn_model.run_validation()


