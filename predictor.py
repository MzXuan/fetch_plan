# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import random
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import attention_decoder as attention_decoder_seq2seq
from tensorflow.contrib.legacy_seq2seq import rnn_decoder
import time, os
import scripts.visual_test_trajs as vt
import csv


class Predictor(object):
    def __init__(self, sess, FLAGS):
        ## extract FLAGS
        self.sess = sess
        
        self.num_units = FLAGS.num_units_cls
        self.num_stacks = FLAGS.num_stacks
        
        self.in_dim = FLAGS.in_dim
        self.out_dim = FLAGS.out_dim
        
        self.in_timesteps = FLAGS.in_timesteps
        self.in_timesteps_min = FLAGS.in_timesteps_min
        self.in_timesteps_max = FLAGS.in_timesteps_max
        self.out_timesteps = FLAGS.out_timesteps
    
        self.validation_interval = FLAGS.validation_interval
        self.checkpoint_interval = FLAGS.checkpoint_interval
        self.sample_interval = FLAGS.sample_interval
        self.display_interval = FLAGS.display_interval
        self.checkpoint_dir = FLAGS.check_dir_cls
        self.sample_dir = FLAGS.sample_dir_cls
        
        self.max_iteration = FLAGS.cls_max_iteration
        self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate
        self.test_type = FLAGS.test_type
        
        self.loss_mode = FLAGS.loss_mode
        self.out_dim_wgts = [FLAGS.out_dim_wgt1, FLAGS.out_dim_wgt2, FLAGS.out_dim_wgt3]
    
        self.run_mode = FLAGS.run_mode

        self.val_traj_num = 3

        self.weight_1 = 1.0
        self.weight_2 = 10.0
        self.gamma = 0.99
        self.gen_samples = 32
        
    
        # ## split datasets
        # self.datasets_list = DATASETS
        # print("the length of datasets: ", len(self.datasets_list))
        # self.datasets = []
        
        ## for record dara
        self.validate_data = 0

        ## build model
        self.build()
    
    
    def calculate_loss(self):
        ## calculate loss as a whole based on all dimensionalities
        if self.loss_mode is 0:
            self.loss = tf.losses.mean_squared_error(self.y_ph, self.mean)

        # if self.loss_mode is 1:
        #     self.mse = tf.square(self.y_ph - self.mean)
        #     self.mse_var = tf.reduce_mean((1.0 / self.std) * self.mse + self.weight_1 * self.std)
        #     self.mse_prob = tf.reduce_mean(tf.square(self.prob - self.prob_ph))
        #     self.loss = self.mse_var + self.weight_2 * self.mse_prob

    def build(self):
        ## define input and output
        self.x_ph = tf.placeholder(tf.float32, shape=[None, self.in_timesteps_max, self.in_dim], name='in_timesteps_max')
        self.x_len_ph = tf.placeholder(tf.int32, shape=[None], name='in_timesteps_len')
        self.y_ph = tf.placeholder(tf.float32, shape=[None, self.out_dim], name='out_timesteps')
    
        ## encoder
        enc_inputs = self.x_ph
        # gru_rnn1 = tf.nn.rnn_cell.GRUCell(64)
        gru_rnn2 = tf.nn.rnn_cell.GRUCell(32)
        gru_rnn3 = tf.nn.rnn_cell.GRUCell(16)
        enc_cell = tf.nn.rnn_cell.MultiRNNCell([gru_rnn2, gru_rnn3])
        enc_outputs, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_inputs, sequence_length=self.x_len_ph, dtype=tf.float32)
        print('enc_outputs shape:', enc_outputs[0].get_shape())
        # print('enc_state shape:', enc_state[0].get_shape())
        # print(enc_state)

        ## dense layer classifier
        dense_outputs = tf.layers.dense(enc_state[0], self.out_dim, activation=tf.nn.sigmoid)
        print('dense shape:', dense_outputs.get_shape())

        self.mean = tf.reshape(dense_outputs, [-1, self.out_dim])
        print('self.pred shape:', self.mean.get_shape())

        ## setup optimization
        self.calculate_loss()
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=tf.trainable_variables())
        print('var_list:\n', tf.trainable_variables())
        
        ## save summary
        tf.summary.scalar('loss', self.loss)
        # tf.summary.scalar('validate', self.validate_data)
        self.merged_summary = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

        ## new a saver for save and load model
        self.saver = tf.train.Saver()


    def pad_input(self, x):
        x_len = len(x)
        padding_data = np.zeros((self.in_timesteps_max - x_len, self.in_dim))
        x_pad = np.concatenate([x, padding_data], axis=0)
        return x_pad


    def feed_one_data(self, end_index, train_x,traj_lens):
        if end_index >= self.in_timesteps_max:
            start_index = np.max(end_index - 50, 0)
            x = train_x[start_index:end_index]
            y = train_x[end_index]
            x_len = end_index - start_index
        else:
            # x = train_x[0:self.in_timesteps_max]
            x = train_x[0:end_index]
            x = self.pad_input(x)
            y = train_x[end_index]
            x_len = end_index

        x = np.asarray(x)
        y = np.asarray(y)

        return x, y, x_len


    def feed_traj_data(self, traj_index):
        datasets = self.datasets_list[traj_index]

        train_x = datasets['traj_x']
        traj_lens = datasets['traj_lens']

        end_index = random.randint(1, traj_lens-1)
        return self.feed_one_data(end_index, train_x,traj_lens)

        
    def feed_data(self):
        xs, ys, x_lens = [], [], []
        traj_nums = len(self.datasets_list)
        for _ in range(self.batch_size):
            traj_index = random.randint(0, traj_nums-self.val_traj_num)
            x, y, x_len = self.feed_traj_data(traj_index)
            xs.append(x)
            ys.append(y)
            x_lens.append(x_len)

        x_lens = np.asarray(x_lens, dtype=np.int32)

        return xs, ys, x_lens

    def feed_data_online(self,x,y,x_len):
        xs, ys, x_lens = [], [], []
        xs.append(x)
        ys.append(y)
        x_lens.append(x_len)

        x_lens = np.asarray(x_lens, dtype=np.int32)

        return xs, ys, x_lens


    def initialize_sess(self):
        ## initialize global variables
        self.sess.run(tf.global_variables_initializer())
        print("initialize model training first")
            

 
    def train_online(self, x, y, x_len,iteration):

        xs,ys,x_lens = self.feed_data_online(x,y,x_len)
        ## run training
        fetches  = [self.train_op, self.merged_summary]
        fetches += [self.loss, self.y_ph, self.mean]
        feed_dict = {self.x_ph:xs, self.y_ph:ys, self.x_len_ph: x_lens}
        _, merged_summary, \
            loss, y, pred = self.sess.run(fetches, feed_dict)
      
        self.file_writer.add_summary(merged_summary, iteration)
        ## save model
        if (iteration % self.checkpoint_interval) is 0:
            self.save(iteration)
            
        ## display information
        if (iteration % self.display_interval) is 0:
            print('\n')
            print('iteration {0}: loss = {1} '.format(
                iteration, loss))

        return loss


    def train(self):
        ## preload the previous saved model or initialize model from scratch
        # path_name = self.load()
        if True:
            ## initialize global variables
            self.sess.run(tf.global_variables_initializer())
            print("initialize model training first")
            
        ## train model
        for iteration in range(self.max_iteration+1):
            ## feed data
            x, y, x_len = self.feed_data()
            ## run training
            fetches  = [self.train_op, self.merged_summary]
            fetches += [self.loss, self.y_ph, self.mean]
            feed_dict = {self.x_ph:x, self.y_ph:y, self.x_len_ph: x_len}
            
            _, merged_summary, \
            loss, y, pred = self.sess.run(fetches, feed_dict)
      
            self.file_writer.add_summary(merged_summary, iteration)
      
            ## validate model
            if  (iteration % self.validation_interval) is 0:
                self.validate()
                validate_summary = tf.Summary()
                validate_summary.value.add(tag="validate rmse", simple_value = self.validate_data)
                self.file_writer.add_summary(validate_summary,iteration)
                # self.file_writer.add_summary(merged_summary, iteration)
                # self.file_writer.add_summary(validate_rmse, iteration)
            ## save model
            if (iteration % self.checkpoint_interval) is 0:
                self.save(iteration)
                
            ## display information
            if (iteration % self.display_interval) is 0:
                print('\n')
                print('iteration {0}: loss = {1} '.format(
                    iteration, loss))

    
    def validate(self):
        print('validating ...')
        rmse_sum = 0
        traj_nums = len(self.datasets_list)
        for i in range(traj_nums-self.val_traj_num-1, traj_nums-1):
            xs, ys, x_lens = [], [], []
            for _ in range(32):
                x, y, x_len = self.feed_traj_data(i)
                xs.append(x)
                ys.append(y)
                x_lens.append(x_len)

            x_lens = np.asarray(x_lens, dtype=np.int32)
            
            fetches = [self.y_ph, self.mean]
            feed_dict = {self.x_ph: xs, self.y_ph: ys,
                         self.x_len_ph: x_lens}
            y, mean = self.sess.run(fetches, feed_dict)

            print('in_timesteps[{0}] y={1}'.format(i, y[-1]))
            print('in_timesteps[{0}] mean={1}'.format(i, mean[-1]))

            rmse = np.sqrt(np.square(y - mean).mean())
            print('validation out_timesteps={0} rmse={1}'.format(self.out_timesteps, rmse))

            rmse_sum += rmse

        rmse_sum = rmse_sum/len(self.datasets_list)
        print('validation all input time_steps rmse_sem = ', rmse_sum)
        self.validate_data=rmse_sum


    def inference(self, x, horizon=50):
        '''
        x shape: (time_step, 7), time_step <= 50
        '''
        # load model first
        self.load()
        x_len = x.shape[0]

        samples = self.gen_samples
        xs = np.tile(np.expand_dims(x, axis=0), (samples, 1, 1))
        
        for _ in range(horizon):
            x_lens = np.tile(x_len, samples)
            if x_len > 50:
                x_len = 50
                xs = xs[:, -50:, :]
            else:
                # padding 0
                padding_data = np.zeros((samples, self.in_timesteps_max-x_len, self.in_dim))
                x_padding = np.concatenate([xs, padding_data], axis=1)

            preds, means = self.sess.run([self.pred, self.mean], feed_dict={
                self.x_ph: x_padding, self.x_len_ph: x_lens
            })

            preds = np.expand_dims(preds, axis=1)
            xs = np.concatenate([xs, preds], axis=1)
            # xs = np.hstack((xs, preds))
            x_len += 1

        return xs


    def test(self):
        # generate x
        xs=[]
        traj_nums = len(self.datasets_list)
        # for i in range(traj_nums - self.val_traj_num - 1, traj_nums - 1):
        i = traj_nums - self.val_traj_num
        datasets = self.datasets_list[i]

        train_x = datasets['traj_x']
        traj_lens = datasets['traj_lens']

        yt=train_x
        for end_index in range(2, traj_lens - 1,5):
            if end_index >= self.in_timesteps_max:
                start_index = np.max(end_index - 50, 0)
                x = train_x[start_index:end_index]
                # y = train_x[end_index]
                # x_len = end_index - start_index
            else:
                x = train_x[0:end_index]

            xs.append(x)

        for samples in xs:
            preds = self.inference(samples)
            # calculate mean and std
            means = np.mean (preds, axis=0)
            stds = np.std(preds, axis = 0)
            trajs = [means, means+stds, means-stds]

            vt.draw_trajs(trajs,  yt)

        #interate inference

        # create trajectory


    def save(self, iteration):
        ## save the model iteratively
        print('iteration {0}: save model to {1}'.format(iteration, self.checkpoint_dir))
        self.saver.save(self.sess, self.checkpoint_dir + '/model.ckpt', global_step=iteration)
    
    
    def load(self):
        ## load the previous saved model
        path_name = tf.train.latest_checkpoint(self.checkpoint_dir)
        if path_name is not None:
            self.saver.restore(self.sess, path_name)
            print('restore model from checkpoint path: ', path_name)
        else:
            print('no checkpoints are existed in ', self.checkpoint_dir)
            
        return path_name
