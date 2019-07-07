# -*- coding: utf-8 -*-
import time, os
import numpy as np
import pickle

class DatasetStru(object):
    def __init__(self, x, x_len, x_mean, x_var, x_start_raw):
        """
        :param x: shape = (self.in_timesteps_max, self.in_dim)
        :param x_len: shape = 1
        :param x_mean: shape = (self.in_dim)
        :param x_var:  shape = (self.in_dim)
        """
        # self.x = np.asarray(x)
        self.x = np.asarray(x)  # input is the delta x after mean and std
        self.x_len = x_len
        self.x_mean = x_mean
        self.x_var = x_var
        self.x_start_raw = x_start_raw


class RLDataCreator():
    # todo: change x to delta x, use new mean, std
    def __init__(self, batch_size):

        self.batch_size = batch_size
        self.in_dim = 10

        self.xs_raw = [[] for _ in range(0, self.batch_size)]
        self.xs = [[] for _ in range(0, self.batch_size)]
        self.x_lens = np.zeros(batch_size, dtype=int)
        self.x_mean = np.zeros(self.in_dim)
        self.x_var = np.zeros(self.in_dim)
        self.dataset = []
        self.dataset_delta = []

        self.collect_flag = False

    #todo: use mean-std method optimize delta x


    def _create_seq(self, obs_raw, dones, infos):
        """
        create sequences from input observations;
        reset sequence if a agent is done its task
        :param obs:  observations from environment
        :param dones: whether the agent is done its task
        :param mean: mean of observations
        :param var: variations of observations
        :return: done sequences
        """
        seqs_done, seqs_all = [], []

        for idx, (ob, done) in enumerate(zip(obs_raw, dones)):
            if done:
                if not infos[idx]['is_collision']:
                    x_seq = np.asarray(self.xs_raw[idx])
                    # create a container saving reseted sequences for future usage
                    seqs_done.append(x_seq)
                else:
                    print("in collision")
                self.xs_raw[idx] = []
                self.x_lens[idx] = 0

            self.xs_raw[idx].append(np.concatenate((ob[6:13],
                                                ob[0:3])))
            self.x_lens[idx] += 1

        return seqs_done

    def _create_traj(self, trajs):
        """
        create dataset from saved sequences
        :return:
        """
        for traj in trajs:
            x_len = traj.shape[0]
            if x_len > 20 and x_len < 200:
                self.dataset.append(traj)

        dataset_length = len(self.dataset)

        # for visualization
        if dataset_length%50 < 2 :
            print("collected dataset length:{}".format(dataset_length))


        # if dataset is large enough, stop collect new data and save
        if dataset_length > 100:
            print("Enough data collected, stop getting new data...")
            self.collect_flag = True


    def get_mean_std(self):
        #calculate the mean and std for of the dataset on each dimension
        temp_list = []
        for data in self.dataset:
           temp_list.extend(data)
        temp_list = np.asarray(temp_list)

        # print("shape of temp list")
        # print(temp_list.shape)

        mean = temp_list.mean(axis = 0)
        std = temp_list.std(axis = 0)

        # print("shape of mean")
        # print(mean.shape)
        # print("shape of std")
        # print(std.shape)



        for data in self.dataset:
            x_normal = (data - mean)/std
            x_len = data.shape[0]
            x_start_raw = data[0,:]

            # print("x_len", x_len)
            # print("x_start_raw", x_start_raw)
            self.dataset_delta.append(DatasetStru(x_normal, x_len,
                                                 mean, std, x_start_raw))

        print("save dataset...")
        pickle.dump(self.dataset_delta,
                    open("./pred/" + "/dataset_rl" + ".pkl", "wb"))
            # print("new x shape:")
            # print(x.shape)
            # print("x")
            # print(x)


    def collect(self, obs_raw, dones, infos):
        """
        function: collect sequence dataset
        :param obs: obs.shape = [batch_size, ob_shape] include joint angle etc.
        :param dones: dones.shape = [batch_size]
        :param mean: mean.shape = [batch_size, ob_shape]
        :param var: var.shape = [batch_size, ob_shape]
        """

        # create input sequence
        seqs_done = self._create_seq(obs_raw, dones, infos)

        # create training dataset for future training
        if len(seqs_done) > 0:
            self._create_traj(seqs_done)

        # post process all the sequences
        return self.collect_flag


    def collect_online(self, obs_raw, dones, infos):
        '''

        :param obs_raw:
        :param dones:
        :param infos:
        :return:
        '''
        #todo: get mean and std from dataset
        # calculate delta x and goals

        return 0



