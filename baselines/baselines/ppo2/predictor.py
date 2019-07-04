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


NUM_UNITS = 50
NUM_LAYERS = 3



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

        ## prepare sequcne containers
        self.xs = [[] for _ in range(0, self.batch_size)]
        self.x_lens = np.zeros(batch_size, dtype=int)
        self.x_mean = np.zeros(self.in_dim)
        self.x_var = np.zeros(self.in_dim)

        self.train_model = KP.TrainRNN(self.batch_size, self.in_dim, self.out_dim,
                                       self.in_timesteps_max, self.out_timesteps, self.num_units, num_layers=self.num_layers, load=load,
                                      model_name=model_name)

        self.inference_model = KP.PredictRNN(1,
                                       self.in_dim, self.out_dim, self.in_timesteps_max, 200, self.num_units, num_layers=self.num_layers,
                                    model_name = model_name)
        if load:
            self.train_model.load_model()

        self.inference_model.load_model()


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


    def run_online_prediction(self, x, goals):
        '''

        :param x:
        :param goals:
        :return:
        '''
        return 0

    def run_validation(self):
        ## load dataset
        self._load_train_set()
        print("trajectory numbers: ", len(self.dataset))
        valid_len = int(self.validate_ratio * len(self.dataset))

        valid_set = self._process_dataset(self.dataset[-valid_len:-1])

        last_traj = []
        # for x in valid_set[0]:

        for idx in range(1,len(valid_set[0]), 10):
            x_full = valid_set[4][idx]

            if idx == 1 or (valid_set[4][idx] is not last_traj):
                print("update to new dataset")
                min_dist_list = []
                goals = utils.GetRandomGoal(self.out_dim)
                goals.append(x_full[-1])
                goal_true = [x_full[-1]]
            last_traj = valid_set[4][idx]


            x = np.expand_dims(valid_set[0][idx], axis=0)
            y = np.expand_dims(valid_set[1][idx], axis=0)
            x_len = valid_set[2][idx]
            x_start = valid_set[3][idx]

            y_pred = self.inference_model.predict(X=x, Y=y)

            # -------calculate minimum distance to true goal-----#
            _, _, min_dist = utils.find_goal(y_pred[0], goal_true)
            min_dist_list.append(min_dist)
            # print("min_dist")
            # print(min_dist)
            # -----find goal based on prediction---#
            goal_pred, goal_idx, _ = utils.find_goal(y_pred[0], goals)

            # ------plot predicted data-----------
            import visualize

            x_s = x[0][0]
            for idx, point in enumerate(x_full):
                if np.linalg.norm(x_s-point) < (1e-6):
                    break


            show_x = np.concatenate((x_full[:idx], x[0]), axis=0)
            show_y = np.concatenate((x_full[:idx], y_pred[0]), axis=0)

            visualize.plot_dof_seqs(show_x, show_y, x_full, goals, goal_pred)  # plot delta result
            visualize.plot_dist(min_dist_list)
            time.sleep(2)




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
        x_full = data.x
        if length > self.in_timesteps_max:
            id_end = id_start + self.in_timesteps_max + self.out_timesteps
        else:
            id_end = length

        x_start = x_seq[0, -3:]

        x = x_seq[id_start:id_start+self.in_timesteps_max, -3:]
        y = x_seq[id_start+self.in_timesteps_max:id_end, -3:]


        x = self._padding(x, self.in_timesteps_max, 0.0)

        return x, y, length, x_start, x_full


    def _accumulate_data(self, delta_x, delta_y, x_start):
        x = delta_x + x_start
        y = delta_y + x_start
        return x, y

    def _padding(self, seq, new_length, my_value=None):
        old_length = len(seq)
        value = np.copy(seq[-1, :])
        if not my_value is None:
            value.fill(my_value)
        value = np.expand_dims(value, axis=0)

        if old_length < new_length:
            for _ in range(old_length, new_length):
                    seq = np.append(seq, value, axis=0)
        return seq

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

        for idx, (ob, done) in enumerate(zip(obs, dones)):
            #-------add end label------------
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
            #-------------------------------------------------

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
            if traj.x_len > self.in_timesteps_max and traj.x_len < 200:
                self.dataset.append(traj)

        # display number of data collected
        dataset_length = len(self.dataset)
        if dataset_length % 100 < 10 :
            print("collected dataset length:{}".format(self.dataset_length))

        # if dataset is large, save it
        if dataset_length > 2000:
            print("save dataset...")
            pickle.dump(self.dataset,
                open("./pred/" + "/dataset_new" + ".pkl", "wb"))
            self.collect_flag = True

    def collect(self, obs, dones, infos, mean=None, var=None):
        """
        function: collect sequence dataset
        :param obs: obs.shape = [batch_size, ob_shape] include joint angle etc.
        :param dones: dones.shape = [batch_size]
        :param mean: mean.shape = [batch_size, ob_shape]
        :param var: var.shape = [batch_size, ob_shape]
        """

        #create input sequence
        seqs_done, _ = self._create_seq(obs, dones, infos, mean, var)

        #create training dataset for future training
        if len(seqs_done) > 0:
            self._create_traj(seqs_done)

        return self.collect_flag

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
        print("dataset length is: ")
        print(len(self.dataset))

        for idx, data in enumerate(self.dataset):
            if idx%10 == 0:
                visualize.plot_3d_eef(data.x)
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--iter', default=0, type=int)
    parser.add_argument('--model_name', default='test1', type=str)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    test_flag=args.test
    out_steps=50

    if not os.path.isdir("./pred"):
        os.mkdir("./pred")

    rnn_model = Predictor(1024, in_max_timestep=30, out_timesteps=out_steps, train_flag=True, epoch=args.epoch,
                          iter_start=args.iter, lr=args.lr, load=args.load,
                          model_name="att_{}_{}_{}_seq_tanh".format(NUM_UNITS, NUM_LAYERS, out_steps))

    # rnn_model.plot_dataset()

    if not test_flag:

        rnn_model.run_training()

    else:
        print("start testing...")
        # plot all the validate data step by step

        # rnn_model.plot_dataset()
        rnn_model.run_validation()

