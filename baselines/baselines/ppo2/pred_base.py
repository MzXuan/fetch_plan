# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import os
import time
import csv
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


class PredBase(object):
    def __init__(self, batch_size, in_max_timestep, out_timesteps, train_flag,
                 step = 1, epoch=20, iter_start=0,
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
        self.validate_ratio = 0.1
        self.num_units = NUM_UNITS
        self.num_layers = NUM_LAYERS
        self.in_dim=3
        self.out_dim=3
        self.load = load
        self.model_name = model_name
        self.step = step

        #  prepare containers
        self.x_mean = np.zeros(self.in_dim)
        self.x_var = np.ones(self.in_dim)

        self._load_train_set()


    def init_model(self, train_model, inference_model):
        self.train_model = train_model
        self.inference_model = inference_model


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


    # def run_online_prediction(self, batched_seqs,\
    #                           last_pred_obs_list, last_pred_result_list,  visial_flags = False):
    #     '''
    #     :param batched_seqs: raw x (batch size * input shape)
    #     :param batched_goals: raw goals / true goal (batch size * goal shape)
    #     :return: reward of this prediction, batch size * 1
    #     '''
    #     pred_obs_list = []
    #     pred_result_list = []
    #     pred_loss_list = []
    #     for idx, seq in enumerate(batched_seqs):
    #         seq = seq[:, -3:]
    #         seq_normal = (seq - self.x_mean) / self.x_var
    #         if (seq_normal.shape[0]-1) < self.in_timesteps_max:
    #         # if sequence is toooo short to predict, set return value to 0
    #             pred_result = np.zeros(3)
    #             pred_state = np.zeros(self.num_units*self.num_layers)
    #             pred_loss_list.append(0.0)
    #
    #         elif (seq_normal.shape[0]-1)%self.in_timesteps_max == 0:
    #         # if sequence is long enough and it is good to reset a new prediction based on true data
    #             x_true_normal = seq_normal[-self.in_timesteps_max-1:-1,:]
    #             y_true = seq[-1, :]
    #
    #             target_seq, encoder_state_value = \
    #                 self.inference_model.get_encoder_latent_state(inputs = np.expand_dims(x_true_normal, axis=0))
    #
    #             pred_result, pred_state = \
    #                 self.inference_model.inference_one_step(target_seq, encoder_state_value)
    #
    #
    #             y_pred = pred_result * self.x_var + self.x_mean
    #
    #             pred_state = np.asarray(pred_state).reshape(self.num_units*self.num_layers)
    #             pred_loss_list.append(np.linalg.norm(y_true-y_pred))
    #
    #         else:
    #         # sequence is long enough and we can continue our previous prediction
    #             y_true = seq[-1, :]
    #             target_seq = last_pred_result_list[idx]
    #             last_state= last_pred_obs_list[idx]
    #
    #
    #             target_seq = target_seq.reshape(1,1,3)
    #             temp=last_state.reshape(pred_flags.num_layers, 1, pred_flags.num_units)
    #             states_value = [temp[0], temp[1]]
    #
    #             # print("target_seq shape {}, ".format(target_seq.shape))
    #             # print("and last state shape{}".format(states_value[0].shape))
    #
    #             pred_result, pred_state = \
    #                 self.inference_model.inference_one_step(target_seq, states_value)
    #
    #             y_pred = pred_result * self.x_var + self.x_mean
    #             pred_state = np.asarray(pred_state).reshape(self.num_units * self.num_layers)
    #             pred_loss_list.append(np.linalg.norm(y_true-y_pred))
    #
    #         pred_obs_list.append(pred_state)
    #         pred_result_list.append(pred_result)
    #
    #     return np.asarray(pred_obs_list), pred_result_list, np.asarray(pred_loss_list)


    def run_online_prediction(self, batched_seqs, batched_goals, \
                              batch_alternative_goals, visial_flags = False):
        '''
        :param batched_seqs: raw x (batch size * input shape)
        :param batched_goals: raw goals / true goal (batch size * goal shape)
        :return: reward of this prediction, batch size * 1
        '''
        rewards = []
        batched_seqs_normal = []
        pred_obs_list = []
        for seq in batched_seqs:
            seq = seq[:, -3:]
            seq_normal = (seq - self.x_mean) / self.x_var
            if seq_normal.shape[0] < self.in_timesteps_max:
                seq_normal = self._padding(seq_normal, self.in_timesteps_max, 0.0)
            else:
                seq_normal = seq_normal[-self.in_timesteps_max:]
            batched_seqs_normal.append(seq_normal)

        batched_seqs_normal = np.asarray(batched_seqs_normal)
        ys_pred, enc_states = self.inference_model.predict(X=batched_seqs_normal)

        # then we restore the origin x and calculate the reward
        n_envs = len(batched_seqs)
        for idx in range(0, n_envs):
            seq = batched_seqs[idx]
            if seq.shape[0] < 5:
                rewards.append(0.1)
                pred_obs_list.append(np.zeros(self.num_units*self.num_layers))
            else:
                raw_y_pred = ys_pred[idx] * self.x_var + self.x_mean
                select_goal, goal_idx, min_dist = utils.find_goal(\
                    raw_y_pred, batch_alternative_goals[idx].reshape((3,3)))

                if np.linalg.norm(select_goal-batched_goals[idx])<1e-7:
                    rewards.append(3.0)

                else:
                    rewards.append(0.1)
                pred_obs_list.append(np.concatenate([enc_states[0][idx],enc_states[1][idx]]))

        # print("rewards: ", rewards)
        rewards = np.asarray(rewards)

        return np.asarray(pred_obs_list), rewards

    def run_validation(self):
        ## load dataset
        self._load_train_set()

        valid_len = int(self.validate_ratio * len(self.dataset))
        valid_set = self._process_dataset(self.dataset[-valid_len:-1])
        print("validate dataset numbers: ", len(valid_set[0]))

        start_id = 1
        last_traj = valid_set[4][start_id]

        # for percentage error saving
        ratio_vals = 0.01 * np.array(range(0, 101, 5))
        errors = []
        errors_x = []

        # for idx in range(start_id, len(valid_set[0]), 10):
        for idx in range(start_id, 700, 10):
            x_full = valid_set[4][idx]

            diff = ((x_full[:10]-last_traj[:10])**2).mean() #for detect change of new trajectory

            if idx == start_id or (diff>1e-5):
                # print("update to new dataset {}, processing".format(idx))
                # saving error of last trajectory
                if idx != start_id:
                    errors.append(np.interp(ratio_vals, ratio, min_dist_list))
                    errors_x.append(np.interp(ratio_vals, ratio, min_dist_list_x))
                # clear container
                min_dist_list = []
                min_dist_list_x = []
                ratio = []

                # prepare of next goal
                goals = utils.GetRandomGoal(self.out_dim)
                goals.append(x_full[-1])
                goal_true = [x_full[-1]]
            last_traj = x_full

            x = np.expand_dims(valid_set[0][idx], axis=0)
            y = np.expand_dims(valid_set[1][idx], axis=0)
            x_len = valid_set[2][idx]
            x_start = valid_set[3][idx]


            y_pred, enc_states = self.inference_model.predict(X=x, Y=y)

            print("x: ", x)
            print("y: ", y)
            print("y_pred: ", y_pred)

            # -------calculate minimum distance to true goal-----#
            _, _, min_dist = utils.find_goal(y_pred[0], goal_true)
            min_dist_list.append(min_dist)

            _, _, min_dist_x = utils.find_goal(x[0], goal_true)
            min_dist_list_x.append(min_dist_x)
            # -----find goal based on prediction---#
            goal_pred, goal_idx, _ = utils.find_goal(y_pred[0], goals)


            x_s = x[0][0]
            for idx, point in enumerate(x_full):
                if np.linalg.norm(x_s - point) < (1e-6):
                    break

            ratio.append(idx/x_len)

            # ------plot predicted data (step by step)-----------
            import visualize
            show_x = np.concatenate((x_full[:idx], x[0]), axis=0)
            visualize.plot_dof_seqs(show_x, y_pred[0], step = self.step, y_true = x_full,
                                    goals= goals, goal_pred = goal_pred)  # plot delta result
            visualize.plot_dist(min_dist_list)
            time.sleep(1)

        # ----------write average error and save result------------

        # errors = np.asarray(errors)
        # error_mean = np.mean(errors, axis=0)
        # err_x_mean = np.asarray(errors_x).mean(axis=0)
        # with open('./pred/errors.csv', 'a', newline='') as csvfile:
        #     spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        #     spamwriter.writerow(error_mean)
        # print("error_mean: ", error_mean)

        # import visualize
        # visualize.plot_avg_err(ratio_vals, error_mean, err_x_mean)


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
                self.x_var = np.ones(self.in_dim)
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
            if idx%500 == 0:
                visualize.plot_3d_eef(data.x)
        plt.show()


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


    def _feed_one_data(self, data, id_start = 0):
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

