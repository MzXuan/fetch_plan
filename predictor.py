# -*- coding: utf-8 -*-
import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import rnn_decoder
import time, os


class Predictor(object):
    def __init__(self, sess, FLAGS, 
                 batch_size, max_timestep, train_flag):
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
        
        self.lr = FLAGS.learning_rate
        
        self.out_dim_wgts = [FLAGS.out_dim_wgt1, 
                             FLAGS.out_dim_wgt2, 
                             FLAGS.out_dim_wgt3]
    
        self.run_mode = FLAGS.run_mode

        self.batch_size = batch_size
    
        ## prepare containers for saving data
        self.xs = np.zeros((batch_size, self.in_timesteps_max, self.in_dim))
        self.ys = np.zeros((batch_size, self.out_dim))
        self.x_lens = np.zeros(batch_size, dtype=int)

        ## build model
        self._build_ph()
        self._build_net()

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
    
    def calculate_batch_loss(self, true, pred):
        error = np.sqrt(np.mean((true-pred)**2,axis=1))
        return error

    def _build_net(self):
        ## encoder
        enc_inputs = self.x_ph
        gru_rnn1 = tf.nn.rnn_cell.GRUCell(32)
        gru_rnn2 = tf.nn.rnn_cell.GRUCell(32)
        enc_cell = tf.nn.rnn_cell.MultiRNNCell([gru_rnn1, gru_rnn2])
        _, enc_state = tf.nn.dynamic_rnn(
            enc_cell, enc_inputs, 
            sequence_length=self.x_len_ph, dtype=tf.float32
            )
        
        ## dense layer classifier
        dense_outputs = tf.layers.dense(
            enc_state[1], self.out_dim, activation=tf.nn.sigmoid
            )
        print('dense shape:', dense_outputs.get_shape())

        self.mean = tf.reshape(dense_outputs, [-1, self.out_dim])
        print('self.pred shape:', self.mean.get_shape())

        ## setup optimization
        self.loss = tf.losses.mean_squared_error(self.y_ph, self.mean)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
            self.loss, var_list=tf.trainable_variables())

        print('var_list:\n', tf.trainable_variables())
        
        ## save summary
        tf.summary.scalar('loss', self.loss)
        self.merged_summary = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(
            self.checkpoint_dir, self.sess.graph
            )

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
            fetches += [self.loss, self.y_ph, self.mean]
            feed_dict = {
                self.x_ph:xs, 
                self.y_ph:ys, 
                self.x_len_ph: x_lens
                }

            _, merged_summary, \
            loss, y, y_hat = self.sess.run(fetches, feed_dict)
            batch_loss = self.calculate_batch_loss(y, y_hat)

            self.file_writer.add_summary(merged_summary, self.iteration)
            self.iteration+=1

            ## save model
            if (self.iteration % self.checkpoint_interval) is 0:
                self.save(self.iteration)

        else:
            fetches = [self.loss, self.y_ph, self.mean]
            feed_dict = {
                self.x_ph: xs,
                self.y_ph:ys,
                self.x_len_ph: x_lens
                }

            loss, y, y_hat = self.sess.run(fetches, feed_dict)
            batch_loss = self.calculate_batch_loss(y, y_hat)

        ## display information
        if (self.iteration % self.display_interval) is 0:
            print('\n')
            print("pred = {}, true goal = {}".format(y_hat, y))
            print('predict loss = {} '.format(loss))

        return batch_loss

    def save(self, iteration):
        ## save the model iteratively
        print('iteration {}: save model to {}'.format(
            iteration, self.checkpoint_dir)
            )
        self.saver.save(self.sess, 
            self.checkpoint_dir + '/model.ckpt', 
            global_step=iteration)
    
    def load(self):
        ## load the previous saved model and extract iteration number
        path_name = tf.train.latest_checkpoint(self.checkpoint_dir)
        iteration = path_name.split("model.ckpt-")
        self.iteration=int(iteration[-1])
        
        if path_name is not None:
            self.saver.restore(self.sess, path_name)
            print('restore model from checkpoint path: ', path_name)
        else:
            print('no checkpoints are existed in ', self.checkpoint_dir)
            
        return path_name


if __name__ == '__main__':
    from flags import flags

    train_flag=False
    FLAGS = flags.FLAGS

    def rand_bools_int_func(n):
        import random
        r = random.getrandbits(n)
        return [bool((r>>i)&1) for i in range(n)]

    with tf.Session() as sess:
        # create and initialize session
        rnn_model = Predictor(sess, FLAGS, 32, 10, 
                              train_flag=train_flag)

        rnn_model.initialize_sess()

        for _ in range(0, 5000):
            #create fake data
            obs = np.random.rand(32, 20)
            dones = rand_bools_int_func(32)
            # run the model
            rnn_model.predict(obs, dones)
