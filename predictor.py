# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import rnn_decoder
import time, os


class Predictor(object):
    def __init__(self, sess, FLAGS, batch_size, max_timestep, train_flag):
        ## extract FLAGS
        self.sess = sess
        self.train_flag = train_flag
        
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
    

        ## prepare containers for saving data
        self.xs = np.zeros((batch_size, self.in_timesteps_max, self.in_dim))
        self.ys = np.zeros((batch_size,self.out_dim))
        self.x_lens = np.zeros(batch_size,dtype=int)

        ## build model
        self.build()
    
    
    def calculate_loss(self):
        ## calculate loss as a whole based on all dimensionalities
        self.loss = tf.losses.mean_squared_error(self.y_ph, self.mean)

    def calculate_batch_loss(self, true, pred):
        error = np.sqrt(np.mean((true-pred)**2,axis=1))
        return error

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
        if self.train_flag is True:
            print("initialize model...")
            self.sess.run(tf.global_variables_initializer())
            self.iteration = 0
            print("done")
            
        else:
            self.load()
            self.iteration = 0

    def reset(self, dones):
        # function: reset the lstm cell of predictor.
        # create new sequences
        for idx, done in enumerate(dones):
            if done is True:
                self.xs[idx] = np.zeros((self.in_timesteps_max, self.in_dim))
                self.ys[idx] = np.zeros(self.out_dim)
                self.x_lens[idx] = 0
        pass

    def create_input_sequence(self,obs,dones):
        self.reset(dones)
        for idx, data in enumerate(obs):
            lens = self.x_lens[idx]
            if lens<self.in_timesteps_max:
                self.xs[idx,lens,:] = np.concatenate((data[0:7],data[14:17]))
                self.ys[idx,:] = data[-4:-1]
                self.x_lens[idx]+=1
            else:
                lens = self.in_timesteps_max-1
                self.xs[idx,:] = np.roll(self.xs[idx,:], -1, axis=0)
                self.xs[idx,lens,:] = np.concatenate((data[0:7],data[14:17]))
                self.ys[idx,:] = data[-4:-1]
        pass


    def predict(self, obs, dones):
        # function: predict the goal position
        # input: 
        # obs.shape = [batch_size, ob_shape] include joint angle etc.
        # dones.shape = [batch_size]
        # return:
        # goal.shape = [batch_size, 3]

        #create input sequence
        self.create_input_sequence(obs,dones)

        xs = self.xs
        ys = self.ys
        x_lens = self.x_lens

        if self.train_flag is True:
            ## run training
            fetches  = [self.train_op, self.merged_summary]
            fetches += [self.loss, self.y_ph, self.mean]
            feed_dict = {self.x_ph:xs, self.y_ph:ys, self.x_len_ph: x_lens}
            _, merged_summary, \
                loss, true, pred = self.sess.run(fetches, feed_dict)

            batch_loss = self.calculate_batch_loss(true, pred)

            self.file_writer.add_summary(merged_summary, self.iteration)
            self.iteration+=1

            ## save model
            if (self.iteration % self.checkpoint_interval) is 0:
                self.save(self.iteration)
            ## display information
            if (self.iteration % self.display_interval) is 0:
                print('\n')
                print("pred = {0}, true goal = {1}".format(pred, true))
                print('training iteration {0}: loss = {1} '.format(self.iteration, loss))
        else:
            fetches = [self.loss,self.y_ph, self.mean]
            feed_dict = {self.x_ph: xs,self.y_ph:ys,
            self.x_len_ph: x_lens}
            loss, true_goal, predict_goal = self.sess.run(fetches, feed_dict)
            batch_loss = self.calculate_batch_loss(true_goal, predict_goal)

            ## display information
            if (self.iteration % self.display_interval) is 0:
                print('\n')
                print("pred = {0}, true goal = {1}".format(predict_goal, true_goal))
                print('predict loss = {0} '.format(loss))

        return batch_loss

    
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
