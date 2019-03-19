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
    def __init__(self, sess, FLAGS, batch_size, max_timestep):
        ## extract FLAGS
        self.sess = sess
        
        self.num_units = FLAGS.num_units_cls
        self.num_stacks = FLAGS.num_stacks
        
        self.in_dim = FLAGS.in_dim
        self.out_dim = FLAGS.out_dim
        
        self.in_timesteps_max = max_timestep
    
        self.validation_interval = FLAGS.validation_interval
        self.checkpoint_interval = FLAGS.checkpoint_interval
        self.sample_interval = FLAGS.sample_interval
        self.display_interval = FLAGS.display_interval
        self.checkpoint_dir = FLAGS.check_dir_cls
        self.sample_dir = FLAGS.sample_dir_cls
        
        self.max_iteration = FLAGS.cls_max_iteration
        
        self.learning_rate = FLAGS.learning_rate
        
        self.out_dim_wgts = [FLAGS.out_dim_wgt1, FLAGS.out_dim_wgt2, FLAGS.out_dim_wgt3]
    
        self.run_mode = FLAGS.run_mode

        self.batch_size = batch_size
        self.weight_1 = 1.0
        self.weight_2 = 10.0
        self.gamma = 0.99
        self.gen_samples = 32
        
        
        ## for record dara
        self.validate_data = 0

        ## build model
        self.build()
    
    
    def calculate_loss(self):
        ## calculate loss as a whole based on all dimensionalities
        self.loss = tf.losses.mean_squared_error(self.y_ph, self.mean)

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
        self.merged_summary = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

        ## new a saver for save and load model
        self.saver = tf.train.Saver()


    def initialize_sess(self):
        ## initialize global variables
        self.sess.run(tf.global_variables_initializer())
        print("initialize model training first")

     def reset(self):
        # function: reset the lstm cell of predictor.
        # create new sequences
        self.pred_xs = np.zeros(batch_size, self.in_timesteps_max, self.in_dim)
        self.train_xs = np.zeros(batch_size,self.in_timesteps_max, self.in_dim)
        self.train_ys = np.zeros(batch_size,self.out_dim)
        pass


    def pad_input(self, x):
        x_len = len(x)
        padding_data = np.zeros((self.in_timesteps_max - x_len, self.in_dim))
        x_pad = np.concatenate([x, padding_data], axis=0)
        return x_pad

        
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


    def create_input_sequence(self,x,y=None):
        if y == None:
        # predict sequence
            self.pred_sequence.append(x)
        else:
        # train sequence
            self.train_sequcne.append(y)


    def train(self, obs, achieved_goal, goal):
        # function: predict the goal position
        # input: 
        # obs.shape = [batch_size, time_step, 7]
        # achieved_goal.shape = [batch_size, time_step, 3]
        # goal.shape = [batch_size, time_step, 3]

        #todo: create input sequence

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
            print("pred = {0}, true goal = {1}".format(pred, y))
            print('iteration {0}: loss = {1} '.format(
                iteration, loss))

        pass


    def predict(self, obs, achieved_goal):
        # function: predict the goal position
        # input: 
        # obs.shape = [batch_size, 7]
        # achieved_goal.shape = [batch_size, 3]
        # return:
        # goal.shape = [batch_size, 3]

        #todo: create input sequence
        fetches = [self.y_ph, self.mean]
        feed_dict = {self.x_ph: xs,
                     self.x_len_ph: x_lens}
        goal = self.sess.run(fetches, feed_dict)

        # print('in_timesteps[{0}] mean={1}'.format(i, mean[-1]))

        return goal
        
    
    # def validate(self):
    #     print('validating ...')
    #     rmse_sum = 0
    #     traj_nums = len(self.datasets_list)
    #     for i in range(traj_nums-self.val_traj_num-1, traj_nums-1):
    #         xs, ys, x_lens = [], [], []
    #         for _ in range(32):
    #             x, y, x_len = self.feed_traj_data(i)
    #             xs.append(x)
    #             ys.append(y)
    #             x_lens.append(x_len)

    #         x_lens = np.asarray(x_lens, dtype=np.int32)
            
    #         fetches = [self.y_ph, self.mean]
    #         feed_dict = {self.x_ph: xs, self.y_ph: ys,
    #                      self.x_len_ph: x_lens}
    #         y, mean = self.sess.run(fetches, feed_dict)

    #         print('in_timesteps[{0}] y={1}'.format(i, y[-1]))
    #         print('in_timesteps[{0}] mean={1}'.format(i, mean[-1]))

    #         rmse = np.sqrt(np.square(y - mean).mean())
    #         print('validation out_timesteps={0} rmse={1}'.format(self.out_timesteps, rmse))

    #         rmse_sum += rmse

    #     rmse_sum = rmse_sum/len(self.datasets_list)
    #     print('validation all input time_steps rmse_sem = ', rmse_sum)
    #     self.validate_data=rmse_sum


    # def inference(self, x, horizon=50):
    #     '''
    #     x shape: (time_step, 7), time_step <= 50
    #     '''
    #     # load model first
    #     self.load()
    #     x_len = x.shape[0]

    #     samples = self.gen_samples
    #     xs = np.tile(np.expand_dims(x, axis=0), (samples, 1, 1))
        
    #     for _ in range(horizon):
    #         x_lens = np.tile(x_len, samples)
    #         if x_len > 50:
    #             x_len = 50
    #             xs = xs[:, -50:, :]
    #         else:
    #             # padding 0
    #             padding_data = np.zeros((samples, self.in_timesteps_max-x_len, self.in_dim))
    #             x_padding = np.concatenate([xs, padding_data], axis=1)

    #         preds, means = self.sess.run([self.pred, self.mean], feed_dict={
    #             self.x_ph: x_padding, self.x_len_ph: x_lens
    #         })

    #         preds = np.expand_dims(preds, axis=1)
    #         xs = np.concatenate([xs, preds], axis=1)
    #         # xs = np.hstack((xs, preds))
    #         x_len += 1

    #     return xs


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
