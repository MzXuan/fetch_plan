# -*- coding: utf-8 -*-
import time, os
import joblib
import pickle
import os

import random
import numpy as np
import tensorflow as tf

# for plot saved dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

class DatasetStru(object):
    def __init__(self,x,y,x_mean,x_var,y_mean,y_var):
        self.x = x
        self.y = y
        self.x_mean = x_mean
        self.x_var = x_var
        self.y_mean = y_mean
        self.y_var = y_var


class Predictor(object):
    def __init__(self, sess, FLAGS, 
                 batch_size, max_timestep, train_flag,
                 reset_flag=True, point="3000"):
        ## extract FLAGS
        self.sess = sess
        self._build_flag(FLAGS)

        self.batch_size = batch_size
        self.in_timesteps_max = max_timestep
        self.in_timesteps_min = 1
        self.train_flag = train_flag
        self.point = point
        self.validata_num = 0.5

        self.iteration = 0
            
        ## prepare sequcne containers
        self.xs = np.zeros((batch_size, self.in_timesteps_max, self.in_dim))
        self.ys = np.zeros((batch_size, self.out_dim))
        self.x_lens = np.zeros(batch_size, dtype=int)
        self.x_mean = np.zeros(self.in_dim)
        self.x_var = np.zeros(self.in_dim)
        self.y_mean = np.zeros(self.out_dim)
        self.y_var = np.zeros(self.out_dim)

        ## prepare containers for saving input dataset
        self.dataset = []
        if reset_flag:
            filelist = [f for f in os.listdir("./model/") if f.endswith(".pkl")]
            for f in filelist:
                os.remove(os.path.join("./model/", f))
        self.dataset_idx=0 # for counting the saved dataset index

        ## build model
        self._build_ph()
        self._build_net()

    def _build_flag(self, FLAGS):
        self.num_units = FLAGS.num_units_cls
        self.num_stacks = FLAGS.num_stacks
        self.model_name = FLAGS.model_name
        
        self.in_dim = FLAGS.in_dim
        self.out_dim = FLAGS.out_dim

        self.validation_interval = FLAGS.validation_interval
        self.checkpoint_interval = FLAGS.checkpoint_interval
        self.sample_interval = FLAGS.sample_interval
        self.display_interval = FLAGS.display_interval
        self.checkpoint_dir = FLAGS.check_dir_cls
        self.sample_dir = FLAGS.sample_dir_cls
        
        self.max_iteration = FLAGS.cls_max_iteration
        
        self.lr = FLAGS.learning_rate
        
        self.out_dim_wgts = [FLAGS.out_dim_wgt1, 
                             FLAGS.out_dim_wgt2, 
                             FLAGS.out_dim_wgt3]
    
        self.run_mode = FLAGS.run_mode

    def _build_ph(self):
        self.x_ph = tf.placeholder(
            tf.float32, 
            shape=[None, self.in_timesteps_max, self.in_dim],
            name='in_timesteps_max')
            
        self.x_len_ph = tf.placeholder(
            tf.int32, 
            shape=[None], 
            name='in_timesteps_len')

        self.y_ph = tf.placeholder(
            tf.float32, 
            shape=[None, self.out_dim], 
            name='out_timesteps')

    def _build_net(self):
        ## encoder
        with tf.variable_scope("predictor"):
            enc_inputs = self.x_ph
            gru_rnn1 = tf.nn.rnn_cell.GRUCell(32)
            gru_rnn2 = tf.nn.rnn_cell.GRUCell(16)
            enc_cell = tf.nn.rnn_cell.MultiRNNCell([gru_rnn1, gru_rnn2])
            _, enc_state = tf.nn.dynamic_rnn(
                enc_cell, enc_inputs, 
                sequence_length=self.x_len_ph, dtype=tf.float32
                )
            
            ## dense layer classifier
            self.y_hat = tf.layers.dense(
                enc_state[1], self.out_dim
                )

        ## setup optimization
        self.loss = tf.losses.mean_squared_error(self.y_ph, self.y_hat)

        var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="predictor"
        )
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
            self.loss, var_list=var_list)
        
        ## save summary
        tf.summary.scalar('loss', self.loss)
        self.merged_summary = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(
            self.checkpoint_dir, self.sess.graph
            )

    def _get_batch_loss(self, y, y_hat):
        error = np.linalg.norm((y-y_hat), axis=1)
        # error = np.sqrt(np.mean((y - y_hat)**2,axis=1))
        return error

    def _reset_seq(self, dones):
        # function: reset the lstm cell of predictor.
        # create new sequences
        for idx, done in enumerate(dones):
            if done:
                self.xs[idx] = np.zeros((self.in_timesteps_max, self.in_dim))
                self.ys[idx] = np.zeros(self.out_dim)
                self.x_lens[idx] = 0

    def _create_training_data(self,dataset):
        xs, ys, x_lens = [], [], []

        for _ in range(0, self.batch_size):
            idx = random.randint(0, len(dataset) - 1)
            data = dataset[idx]
            # xs.append(data.x[:,-3:])
            xs.append(data.x)
            ys.append(data.y)
            x_lens.append(len(data.x))
        return xs,ys,x_lens

    def _create_seq(self, obs, dones, mean, var):
        if mean is not None and var is not None:
            ## save mean and var
            self.x_mean = np.concatenate((mean[6:13],
                                        mean[0:3])) #(joint angle, end-effector position)
            self.x_var = np.concatenate((var[6:13],
                                          var[0:3]))
            self.y_mean = mean[3:6]
            self.y_var = var[3:6]

        ## create sequence data
        for idx, data in enumerate(obs):
            lens = self.x_lens[idx]
            if lens < self.in_timesteps_max:
                self.xs[idx, lens, :] = np.concatenate((data[6:13],
                                                        data[0:3]))
                self.ys[idx, :] = data[3:6]
                self.x_lens[idx] += 1
            else:
                lens = self.in_timesteps_max-1
                self.xs[idx, :] = np.roll(self.xs[idx,:], -1, axis=0)
                self.xs[idx, lens, :] = np.concatenate((data[6:13],
                                                        data[0:3]))
                self.ys[idx, :] = data[3:6]

        self._reset_seq(dones)

    def init_sess(self):
        ## initialize global variables
        if self.train_flag:
            # delete existing and dataset to avoid overlap problem
            # initialize new model
            self.sess.run(tf.global_variables_initializer())

        else:
            try:
                self.load_net(("./model/"+self.model_name+"/{}").format(
                    self.point
                ))
            except:
                self.sess.run(tf.global_variables_initializer())

    def create_dataset(self):
        # function: predict the goal position
        # input:
        # obs.shape = [batch_size, ob_shape] include joint angle etc.
        # dones.shape = [batch_size]
        #
        # create dataset
        for idx, length in enumerate(self.x_lens):
            if length > self.in_timesteps_min:
                self.dataset.append(DatasetStru(self.xs[idx], self.ys[idx],
                                                self.x_mean, self.x_var,
                                                self.y_mean, self.y_var))

        # if dataset is large, save it
        if len(self.dataset) > 150000:
            print("save dataset...")
            pickle.dump(self.dataset, open("./model/"
                                           +"/dataset"+str(self.dataset_idx)+".pkl","wb"))
            self.dataset_idx+=1
            self.dataset=[]

    def save_dataset(self):
        # check whether in training process
        if self.train_flag is not True:
            print("Not in training process, saving failed")
            return 0
        else:
            pickle.dump(self.dataset, open("./model/"
                                            +"/dataset"+str(self.dataset_idx)+".pkl", "wb"))
            print("saving dataset successfully")
            self.dataset = []

    def load_dataset(self, file_name):
        ## load dataset

        try:
            self.dataset = pickle.load(open(os.path.join("./model/", file_name), "rb"))
            # random.shuffle(self.dataset)
        except:
            print("Can not load dataset. Please first run the training stage to save dataset.")
            return 0

        return 1

    def _revert_data(self,data,mean,var):
        return(data*(var+1e-8)+mean)


    def run_training(self):
        #function: train the model according to saved dataset
        import visualize

        ## check whether in training
        if not self.train_flag:
            print("Not in training process,return...")
            return 0

        ## check saved data set
        filelist = [f for f in os.listdir("./model/") if f.endswith(".pkl")]
        num_sets = len(filelist)-1
        self.dataset_idx = 0


        ## prepare threshold to switch dataset
        max_iteration = int(self.point)
        iter_range = range(0,max_iteration,500)
        iter_idx=0
        print("iter_range")
        print(iter_range)
        ## run training
        while self.iteration<max_iteration:
            #----- load dataset ----------#
            if iter_idx < len(iter_range):
                if self.iteration==iter_range[iter_idx]:
                    print("switch to...{}".format(filelist[self.dataset_idx]))
                    iter_idx+=1
                    if self.load_dataset(filelist[self.dataset_idx]) == 0:
                        return 0
                    self.dataset_idx+=1
                    if self.dataset_idx >= num_sets:
                        self.dataset_idx = 0
            #-----create training data----#
            train_num=int(self.validata_num*len(self.dataset))
            xs,ys,x_lens=self._create_training_data(self.dataset)
            #----start training-----#
            fetches = [self.train_op, self.merged_summary]
            fetches += [self.loss, self.y_ph, self.y_hat]
            feed_dict = {
                self.x_ph: xs,
                self.y_ph: ys,
                self.x_len_ph: x_lens
            }

            _, merged_summary, \
            loss, y, y_hat = self.sess.run(fetches, feed_dict)
            batch_loss = self._get_batch_loss(y, y_hat)

            self.file_writer.add_summary(merged_summary, self.iteration)
            self.iteration += 1

            # save model
            if (self.iteration % self.checkpoint_interval) is 0:
                self.save_net(("./model/" + self.model_name + "/{}").format(
                    self.iteration
                ))

            # ## display information
            # if (self.iteration % self.display_interval) is 0:
            #     print('\n')
            #     print("pred = {}, true goal = {}".format(y_hat[0], y[0]))
            #     print('iteration = {}, training loss = {} '.format(self.iteration,loss))


            #----------validate process--------#
            ## validate model
            if (self.iteration % self.validation_interval) is 0:
                print("load validate dataset {}".format(filelist[-1]))
                validate_set = \
                    pickle.load(open(os.path.join("./model/", filelist[-1]), "rb"))

                ## create validate data
                xs, ys, x_lens = self._create_training_data(validate_set)

                ## run validation
                fetches = [self.loss, self.x_ph, self.y_ph, self.y_hat]
                feed_dict = {
                    self.x_ph: xs,
                    self.y_ph: ys,
                    self.x_len_ph: x_lens
                }
                loss, x, y, y_hat = self.sess.run(fetches, feed_dict)

                ## write summary
                validate_summary = tf.Summary()
                validate_summary.value.add(tag="validate rmse", simple_value=loss)
                self.file_writer.add_summary(validate_summary, self.iteration)

                ## display
                # visualize.plot_3d_pred(x[0], y[0], y_hat[0])


                if (self.iteration % self.display_interval) is 0:
                    mean_y = self.dataset[0].y_mean
                    var_y = self.dataset[0].y_var
                    y_origin = self._revert_data(y[0],mean_y,var_y)
                    y_hat_origin = self._revert_data(y_hat[0],mean_y,var_y)
                    print('\n')
                    # print("x = {}".format(x[0]))
                    print("pred = {}, true goal = {}".format(y[0], y_hat[0]))
                    print('iteration = {}, validate loss = {} '.format(self.iteration, loss))
            #---------create validate data-----#

        print("finish training")


    def predict(self, obs, dones, mean=None, var=None):
        # function: predict the goal position
        # input: 
        # obs.shape = [batch_size, ob_shape] include joint angle etc.
        # dones.shape = [batch_size]
        # mean.shape = [batch_size, ob_shape]
        # var.shape = [batch_size, ob_shape]
        # return:
        # batch_loss.shape = [batch_size]

        #create input sequence
        self._create_seq(obs, dones, mean, var)

        # #---- plot created dataset
        # import visualize
        # visualize.plot_3d_pred(self.xs[0],self.ys[0])
        # #----------------------------------
        xs = self.xs
        ys = self.ys
        x_lens = self.x_lens

        if self.train_flag:
            #----create training dataset for future training---#
            self.create_dataset()
            return np.zeros((len(dones)))
            # =========================================
        else:
            #---predict input data---#
            fetches = [self.loss, self.y_ph, self.y_hat]
            feed_dict = {
                self.x_ph: xs,
                self.y_ph:ys,
                self.x_len_ph: x_lens
                }

            loss, y, y_hat = self.sess.run(fetches, feed_dict)


            batch_loss = self._get_batch_loss(y, y_hat)



            # ## display information
            # if (self.iteration % self.display_interval) is 0:
            #     print('\n')
            #     print("pred = {}, true goal = {}".format(y_hat[0], y[0]))
            #     print('predict loss = {} '.format(loss))

            # #------plot predicted data-----------
            # import visualize
            # visualize.plot_3d_pred(xs[0],y[0],y_hat[0])
            # #------------------------------------#
            # print("batch loss")
            # print(batch_loss)

            return batch_loss


    def save_net(self, save_path):
        params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="predictor"
        )
        ps = self.sess.run(params)

        directory = os.path.dirname(save_path)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        joblib.dump(ps, save_path)

    def load_net(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="predictor"
        )

        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)


def plot_dataset(dataset):
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
    # print robot trajectory
    for i, data in enumerate(dataset[5:6]):
        x = revert_data(data.x, data.x_mean, data.x_var)
        y = revert_data(data.y, data.y_mean, data.y_var)

        ax.plot(x[:, -3], x[:, -2], x[:, -1], '-o', linewidth=2, color='black', label="x")
        ax.plot([y[0]], [y[1]], [y[2]], marker='o', markersize=3, linewidth=2, color='red')
    plt.show()
    # for i, data in enumerate(dataset):
    #     if i%8001 is 0:
    #         x=revert_data(data.x,data.x_mean,data.x_var)
    #         y=revert_data(data.y,data.y_mean,data.y_var)
    #
    #         ax.plot(x[:, -3], x[:, -2], x[:, -1], '-o', linewidth=2, color='black', label="x")
    #         ax.plot([y[0]], [y[1]], [y[2]], marker='o', markersize=3, linewidth=2, color='red')
    # plt.show()

def revert_data(data, mean, var):
    return(data*(var+1e-8)+mean)


if __name__ == '__main__':
    from flags import flags

    train_flag=True
    FLAGS = flags.FLAGS

    def rand_bools_int_func(n):
        import random
        r = random.getrandbits(n)
        return [bool((r>>i)&1) for i in range(n)]

    with tf.Session() as sess:
        if train_flag:
            # create and initialize session
            rnn_model = Predictor(sess, FLAGS, 16, 30,
                                  train_flag=True, reset_flag=False)

            rnn_model.init_sess()
            # for _ in range(5000):
            #     #create fake data
            #     obs = np.random.rand(32, 20)
            #     dones = rand_bools_int_func(32)
            #     # run the model
            #     rnn_model.predict(obs, dones)

            # rnn_model.save_dataset()
            rnn_model.run_training()

            # # plot saved dataset
            # filelist = [f for f in os.listdir("./model/human_predict_test") if f.endswith(".pkl")]
            # if rnn_model.load_dataset(filelist[0]) is not 0:
            #     dataset=rnn_model.dataset
            # plot_dataset(dataset)

        else:
            #plot all the validate data step by step
            rnn_model = Predictor(sess, FLAGS, 16, 30,
                                  train_flag=False, reset_flag=False)

            rnn_model.init_sess()





