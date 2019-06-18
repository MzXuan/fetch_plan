# -*- coding: utf-8 -*-
import time, os
import numpy as np
import pickle

class DatasetStru(object):
    def __init__(self, x, x_len, x_mean, x_var):
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


class RLDataCreator():
    # todo: change x to delta x, use new mean, std
    def init(self, batch_size):

        self.batch_size = batch_size
        self.in_dim = 10

        self.xs = [[] for _ in range(0, self.batch_size)]
        self.x_lens = np.zeros(batch_size, dtype=int)
        self.x_mean = np.zeros(self.in_dim)
        self.x_var = np.zeros(self.in_dim)

        self.dataset = []

    #     use mean-std method optimize delta x


    def _create_seq(self, obs, dones, infos, mean, var):
        """
        create sequences from input observations;
        reset sequence if a agent is done its task
        :param obs:  observations from environment
        :param dones: whether the agent is done its task
        :param mean: mean of observations
        :param var: variations of observations
        :return: done sequences
        """
        if mean is not None and var is not None:
            ## save mean and var
            self.x_mean = np.concatenate((mean[6:13],
                                          mean[0:3]))
            self.x_var = np.concatenate((var[6:13],
                                         var[0:3]))

        seqs_done, seqs_all = [], []
        seqs_done_origin = []

        for idx, (ob, done) in enumerate(zip(obs, dones)):
            if done:
                if not infos[idx]['is_collision']:
                    # create a container saving reseted sequences for future usage
                    seqs_done.append(DatasetStru(self.xs[idx], self.x_lens[idx],
                                                 self.x_mean, self.x_var))
                else:
                    print("in collision")
                self.xs[idx] = []
                self.x_lens[idx] = 0

            self.xs[idx].append(np.concatenate((ob[6:13],
                                                ob[0:3])))
            # -------------------------------------------------

            self.x_lens[idx] += 1
            seqs_all.append(DatasetStru(self.xs[idx], self.x_lens[idx],
                                        self.x_mean, self.x_var))

        return seqs_done, seqs_all

    def _create_traj(self, trajs):
        """
        create dataset from saved sequences
        :return:
        """
        for traj in trajs:
            if traj.x_len > 20 and traj.x_len < 300:
                self.dataset.append(traj)
                dataset_length = len(self.dataset)

        # for visualization
        if dataset_length%100 < 10 :
            print("collected dataset length:{}".format(dataset_length))

        # if dataset is large, save it
        if dataset_length > 1000:
            print("save dataset...")
            pickle.dump(self.dataset,
                open("./pred/" + "/dataset_rl" + ".pkl", "wb"))
            self.collect_flag = True

    def collect(self, obs, dones, infos, mean=None, var=None):
        """
        function: collect sequence dataset
        :param obs: obs.shape = [batch_size, ob_shape] include joint angle etc.
        :param dones: dones.shape = [batch_size]
        :param mean: mean.shape = [batch_size, ob_shape]
        :param var: var.shape = [batch_size, ob_shape]
        """

        # create input sequence
        seqs_done, _ = self._create_seq(obs, dones, infos, mean, var)

        # create training dataset for future training
        if len(seqs_done) > 0:
            self._create_traj(seqs_done)

        # print("dataset length: ", self.dataset_length)
        return self.collect_flag



