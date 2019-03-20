# -*- coding: utf-8 -*-
import time, os
import joblib
import random

import numpy as np
import tensorflow as tf


class Predictor(object):
    def __init__(self, sess, FLAGS, 
                 batch_size, max_timestep, train_flag,
                 point="300"):
        ## extract FLAGS
        self.sess = sess
        self._build_flag(FLAGS)

        self.batch_size = batch_size
        self.in_timesteps_max = max_timestep
        self.train_flag = train_flag
        self.point = point

        self.iteration = 0
            
        ## prepare containers for saving data
        self.xs = np.zeros((batch_size, self.in_timesteps_max, self.in_dim))
        self.ys = np.zeros((batch_size, self.out_dim))
        self.x_lens = np.zeros(batch_size, dtype=int)

        ## build model
        self._build_ph()
        self._build_net()

    def _build_flag(self, FLAGS):
        self.num_units = FLAGS.num_units_cls
        self.num_stacks = FLAGS.num_stacks
        
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
            gru_rnn2 = tf.nn.rnn_cell.GRUCell(32)
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
        error = np.sqrt(np.mean((y - y_hat)**2,axis=1))
        return error

    def _reset_seq(self, dones):
        # function: reset the lstm cell of predictor.
        # create new sequences
        for idx, done in enumerate(dones):
            if done is True:
                self.xs[idx] = np.zeros((self.in_timesteps_max, self.in_dim))
                self.ys[idx] = np.zeros(self.out_dim)
                self.x_lens[idx] = 0

    def _create_seq(self, obs, dones):
        self._reset_seq(dones)
        for idx, data in enumerate(obs):
            lens = self.x_lens[idx]
            if lens < self.in_timesteps_max:
                self.xs[idx, lens, :] = np.concatenate((data[0:7], 
                                                        data[14:17]))
                self.ys[idx, :] = data[-3:]
                self.x_lens[idx] += 1
            else:
                lens = self.in_timesteps_max-1
                self.xs[idx, :] = np.roll(self.xs[idx,:], -1, axis=0)
                self.xs[idx, lens, :] = np.concatenate((data[0:7], 
                                                        data[14:17]))
                self.ys[idx, :] = data[-3:]

    def init_sess(self):
        ## initialize global variables
        if self.train_flag:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.load_net("./model/human_predict_test/{}".format(
                self.point
            ))
            
    def predict(self, obs, dones):
        # function: predict the goal position
        # input: 
        # obs.shape = [batch_size, ob_shape] include joint angle etc.
        # dones.shape = [batch_size]
        # return:
        # batch_loss.shape = [batch_size]

        #create input sequence
        self._create_seq(obs, dones)

        xs = self.xs
        ys = self.ys
        x_lens = self.x_lens

        if self.train_flag:
            ## run training
            fetches  = [self.train_op, self.merged_summary]
            fetches += [self.loss, self.y_ph, self.y_hat]
            feed_dict = {
                self.x_ph:xs, 
                self.y_ph:ys, 
                self.x_len_ph: x_lens
                }

            _, merged_summary, \
            loss, y, y_hat = self.sess.run(fetches, feed_dict)
            batch_loss = self._get_batch_loss(y, y_hat)

            self.file_writer.add_summary(merged_summary, self.iteration)
            self.iteration+=1

            ## save model
            if (self.iteration % self.checkpoint_interval) is 0:
                self.save_net("./model/human_predict_test/{}".format(
                    self.iteration
                ))

        else:
            fetches = [self.loss, self.y_ph, self.y_hat]
            feed_dict = {
                self.x_ph: xs,
                self.y_ph:ys,
                self.x_len_ph: x_lens
                }

            loss, y, y_hat = self.sess.run(fetches, feed_dict)
            batch_loss = self._get_batch_loss(y, y_hat)

        ## display information
        if (self.iteration % self.display_interval) is 0:
            print('\n')
            print("pred = {}, true goal = {}".format(y_hat[0], y[0]))
            print('predict loss = {} '.format(loss))

        return batch_loss

    def save_net(self, save_path):
        params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="predictor"
        )
        ps = self.sess.run(params)
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

if __name__ == '__main__':
    from flags import flags

    train_flag=True
    FLAGS = flags.FLAGS

    def rand_bools_int_func(n):
        import random
        r = random.getrandbits(n)
        return [bool((r>>i)&1) for i in range(n)]

    with tf.Session() as sess:
        # create and initialize session
        rnn_model = Predictor(sess, FLAGS, 32, 10, 
                              train_flag=train_flag)

        rnn_model.init_sess()
        for _ in range(0, 5000):
            #create fake data
            obs = np.random.rand(32, 20)
            dones = rand_bools_int_func(32)
            # run the model
            rnn_model.predict(obs, dones)
