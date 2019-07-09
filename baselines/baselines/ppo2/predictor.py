# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import os
import time
import numpy as np

import visualize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

import utils
# import keras_predictor as KP
import keras_seq2seq as KP
from create_traj_set import DatasetStru
from create_traj_set import RLDataCreator

import pred_flags

NUM_UNITS = pred_flags.num_units
NUM_LAYERS = pred_flags.num_layers



class Predictor(object):
    def __init__(self, batch_size, in_max_timestep, out_timesteps, train_flag,
                 epoch=20, iter_start=0,
                 lr=0.001, load=False, model_name="test"):
        ## extract FLAGS
        if iter_start == 0 and lr<0.001:
            self.start_iter = 20
        else:
            self.start_iter = iter_start * epoch
        self.iteration = 0
        self.dataset_length = 0

        self.batch_size = batch_size
        self.in_timesteps_max = in_max_timestep
        self.out_timesteps = out_timesteps

        self.train_flag = train_flag
        self.epochs = epoch
        self.lr = lr
        self.validate_ratio = 0.2
        self.num_units = NUM_UNITS
        self.num_layers = NUM_LAYERS
        self.in_dim=3
        self.out_dim=3

        self.step = 5

        #  prepare containers
        self.x_mean = np.zeros(self.in_dim)
        self.x_var = np.zeros(self.in_dim)

        self.train_model = KP.TrainRNN(self.batch_size,
                                       self.in_dim, self.out_dim, self.out_timesteps, self.num_units, num_layers=self.num_layers, load=load,
                                      model_name=model_name)

        self.inference_model = KP.PredictRNN(self.batch_size,
                                       self.in_dim, self.out_dim, self.in_timesteps_max, 3*self.out_timesteps, self.num_units, num_layers=self.num_layers,
                                    model_name = model_name)

        if load:
            self.train_model.load_model()

        self.inference_model.load_model()

        self._load_train_set()


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

        x_set = train_set[0]
        y_set = train_set[1]

        self.train_model.training(X=x_set, Y=y_set, epochs=self.epochs)


    def run_online_prediction(self, batched_seqs, batched_goals, visial_flags = False):
        '''
        :param batched_seqs: raw x (batch size * input shape)
        :param batched_goals: raw goals / true goal (batch size * goal shape)
        :return: reward of this prediction, batch size * 1
        '''
        # todo: rewrite this to fit batch size
        rewards = []
        batched_seqs_normal = []
        for seq in batched_seqs:
            seq = seq[:, -3:]
            seq_normal = (seq - self.x_mean) / self.x_var
            if seq_normal.shape[0] < self.in_timesteps_max:
                seq_normal = self._padding(seq_normal, self.in_timesteps_max, 0.0)
            else:
                seq_normal = seq_normal[-self.in_timesteps_max:]
            batched_seqs_normal.append(seq_normal)

        batched_seqs_normal = np.asarray(batched_seqs_normal)
        _, ys_pred = self.inference_model.predict(X=batched_seqs_normal)

        # then we restore the origin x and calculate the reward
        for seq, y_pred, goal in zip(batched_seqs, ys_pred, batched_goals):
            if seq.shape[0] < 5:
                rewards.append(0.0)
            else:
                raw_y_pred = y_pred * self.x_var + self.x_mean
                _, _, min_dist = utils.find_goal(raw_y_pred, [goal])
                rewards.append(min_dist)

        # print("rewards: ", rewards)
        rewards = np.asarray(rewards)
        return rewards


        # rewards = []
        # # preprocess batched seqs:
        # for seq, goal in zip(batched_seqs, batched_goals):
        #     seq = seq[:,-3:]
        #     if seq.shape[0] < 5:
        #         # print("seq.shape:", seq.shape)
        #         rewards.append(0.0)
        #         self.min_dist_list = []
        #     else:
        #         seq_normal = (seq - self.x_mean) / self.x_var
        #         if seq_normal.shape[0] < self.in_timesteps_max:
        #             seq_normal = self._padding(seq_normal, self.in_timesteps_max, 0.0)
        #         else:
        #             seq_normal = seq_normal[-self.in_timesteps_max:]
        #
        #         #predict
        #         seq_normal = np.expand_dims(seq_normal, axis=0)
        #         _, y_pred = self.inference_model.predict(X=seq_normal)
        #
        #         # then we restore the origin x and calculate the reward
        #         raw_y_pred = y_pred * self.x_var + self.x_mean
        #         _, _, min_dist = utils.find_goal(raw_y_pred[0], [goal])
        #
        #         rewards.append(min_dist)
        #
        #         # ------plot predicted data-----------
        #         if visial_flags is True:
        #             import visualize
        #             show_y = np.concatenate((seq, raw_y_pred[0]), axis=0)
        #             self.min_dist_list.append(min_dist)
        #             visualize.plot_dof_seqs(x=seq, y_pred = show_y, goals = [goal])  # plot delta result
        #             visualize.plot_dist(self.min_dist_list)
        #             time.sleep(0.1)
        #
        # # print("rewards: ", rewards)
        # rewards = np.asarray(rewards)
        # return rewards

    def run_validation(self):
        ## load dataset
        self._load_train_set()
        print("trajectory numbers: ", len(self.dataset))
        valid_len = int(self.validate_ratio * len(self.dataset))

        valid_set = self._process_dataset(self.dataset[-valid_len:-1])

        start_id = 170
        last_traj = valid_set[4][start_id]

        for idx in range(start_id, len(valid_set[0]), 10):
            x_full = valid_set[4][idx]

            diff = ((x_full[:10]-last_traj[:10])**2).mean() #for detect change of new trajectory

            if idx == start_id or (diff>1e-5):
                print("update to new dataset")
                min_dist_list = []
                goals = utils.GetRandomGoal(self.out_dim)
                goals.append(x_full[-1])
                goal_true = [x_full[-1]]
            last_traj = x_full

            x = np.expand_dims(valid_set[0][idx], axis=0)
            y = np.expand_dims(valid_set[1][idx], axis=0)
            x_len = valid_set[2][idx]
            x_start = valid_set[3][idx]

            _, y_pred = self.inference_model.predict(X=x, Y=y)

            # -------calculate minimum distance to true goal-----#
            _, _, min_dist = utils.find_goal(y_pred[0], goal_true)
            min_dist_list.append(min_dist)
            # -----find goal based on prediction---#
            goal_pred, goal_idx, _ = utils.find_goal(y_pred[0], goals)

            # ------plot predicted data-----------
            import visualize

            x_s = x[0][0]
            for idx, point in enumerate(x_full):
                if np.linalg.norm(x_s - point) < (1e-6):
                    break

            show_x = np.concatenate((x_full[:idx], x[0]), axis=0)


            visualize.plot_dof_seqs(show_x, y_pred[0], step = self.step, y_true = x_full,
                                    goals= goals, goal_pred = goal_pred)  # plot delta result
            visualize.plot_dist(min_dist_list)
            time.sleep(1)


    def _process_dataset(self, trajs):
        xs, ys, x_lens, xs_start, xs_full = [], [], [], [], []
        for traj in trajs:
            for t_start in range(0, traj.x_len-self.in_timesteps_max-self.out_timesteps):
                x, y, x_len, x_start, x_full = self._feed_one_data(traj, t_start)
                xs.append(x)
                ys.append(y)
                x_lens.append(x_len)
                xs_start.append(x_start)
                xs_full.append(x_full)

        xs=np.asarray(xs, dtype=np.float32)
        ys=np.asarray(ys)
        x_lens=np.asarray(x_lens, dtype=np.int32)
        xs_start=np.asarray(xs_start)
        return [xs, ys, x_lens, xs_start, xs_full]



    def _feed_one_data(self, data, id_start = 0, step=5):
        # X: N step; Y: N+M step
        length = data.x_len
        x_seq = data.x
        x_full = data.x[:,-3:]

        step = self.step


        if length > self.in_timesteps_max:
            id_end = id_start + self.in_timesteps_max + step*self.out_timesteps
        else:
            id_end = length

        x_start = x_seq[0, -3:]

        x = x_seq[id_start:id_start+self.in_timesteps_max, -3:]
        y = x_seq[id_start+self.in_timesteps_max:id_end:step, -3:]


        x = self._padding(x, self.in_timesteps_max, 0.0)
        y = self._padding(y, self.out_timesteps)


        return x, y, length, x_start, x_full

    # def _feed_one_data(self, data, id_start = 0):
    #     # X: N step; Y: N+M step
    #     length = data.x_len
    #     x_seq = data.x
    #     x_full = data.x[:,-3:]
    #
    #     if length > id_start + self.in_timesteps_max + self.out_timesteps:
    #         id_end = id_start + self.in_timesteps_max + self.out_timesteps
    #     else:
    #         id_end = length
    #
    #     x_start = x_seq[0, -3:]
    #
    #     x = x_seq[id_start:id_start+self.in_timesteps_max, -3:]
    #     y = x_seq[id_start+self.in_timesteps_max:id_end, -3:]
    #
    #
    #     x = self._padding(x, self.in_timesteps_max, 0.0)
    #     y = self._padding(y, self.out_timesteps)
    #
    #     return x, y, length, x_start, x_full


    def _accumulate_data(self, delta_x, delta_y, x_start):
        x = delta_x + x_start
        y = delta_y + x_start
        return x, y

    def _padding(self, seq, new_length, my_value=None):
        old_length = len(seq)
        if old_length < new_length:
            value = np.zeros((seq.shape[-1]))
            if not my_value is None:
                value.fill(my_value)
            else:
                value = np.copy(seq[-1, :])
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
        filelist = [f for f in os.listdir("./pred/") if f.endswith("rl.pkl")]
        num_sets = len(filelist)

        # NOTICE: we can only use one dataset here because of mean and var problem
        self.dataset = []
        for idx in range(num_sets):
            dataset = self.load_dataset(filelist[idx])
            if dataset == 0:
                self.x_mean = np.zeros(self.in_dim)
                self.x_var = np.zeros(self.in_dim)
                return 0
            else:
                self.dataset.extend(dataset)
                self.x_mean = dataset[0].x_mean[-3:]
                self.x_var = dataset[0].x_var[-3:]

    def plot_dataset(self):
        self._load_train_set()
        #plot dataset
        print("dataset length is: ")
        print(len(self.dataset))

        for idx, data in enumerate(self.dataset):
            if idx%10 == 0:
                visualize.plot_3d_eef(data.x)
        plt.show()


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

        rnn_model = Predictor(1024, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=out_steps,
                              train_flag=True, epoch=args.epoch,
                              iter_start=args.iter, lr=args.lr, load=args.load,
                              model_name=pred_flags.model_name)

        rnn_model.run_training()

    else:

        rnn_model = Predictor(1, in_max_timestep=pred_flags.in_timesteps_max, out_timesteps=out_steps,
                              train_flag=True, epoch=args.epoch,
                              iter_start=args.iter, lr=args.lr, load=args.load,
                              model_name=pred_flags.model_name)
        print("start testing...")
        # plot all the validate data step by step

        # rnn_model.plot_dataset()
        rnn_model.run_validation()

