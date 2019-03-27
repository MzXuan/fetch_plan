# -*- coding: utf-8 -*-
import time, os
import joblib
import pickle
import os

import random
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.contrib.seq2seq import BasicDecoder, TrainingHelper

# for plot saved dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

class DatasetStru(object):
    def __init__(self, x, x_len, x_mean, x_var):
        """

        :param x: shape = (self.in_timesteps_max, self.in_dim)
        :param x_len: shape = 1
        :param x_mean: shape = (self.in_dim)
        :param x_var:  shape = (self.in_dim)
        """
        self.x = np.asarray(x)
        self.x_len = x_len
        self.x_mean = x_mean
        self.x_var = x_var

    def padding(self, seq, new_length):
        old_length = len(seq)
        value = seq[-1,:]
        value = np.expand_dims(value, axis=0)
        for _ in range(old_length,new_length):
            seq=np.append(seq, value, axis=0)
        return seq



class Predictor(object):
    def __init__(self, sess, FLAGS, 
                 batch_size, max_timestep, train_flag,
                 reset_flag=True, point="20000"):
        ## extract FLAGS
        self.sess = sess
        self._build_flag(FLAGS)

        self.batch_size = batch_size
        self.in_timesteps_max = max_timestep
        self.out_timesteps = 10
        self.train_flag = train_flag
        self.point = point
        self.validata_num = 0.5

        self.iteration = 0
            
        ## prepare sequcne containers
        # self.xs = np.zeros((batch_size, self.in_timesteps_max, self.in_dim))
        self.xs = [[] for _ in range(0,self.batch_size)]
        self.x_lens = np.zeros(batch_size, dtype=int)
        self.x_mean = np.zeros(self.in_dim)
        self.x_var = np.zeros(self.in_dim)

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

        self.y_train = tf.placeholder(
            tf.float32,
            shape=[None, self.out_timesteps, self.out_dim],
            name='out_timesteps')


        self.y_ph = tf.placeholder(
            tf.float32, 
            shape=[None, self.out_timesteps, self.out_dim],
            name='out_timesteps')

        self.decoder_seq_length = tf.placeholder(
            tf.int32, shape=[None], name='batch_seq_length'
        )

        self.go_token = np.full((self.out_dim),1.0, dtype=np.float32)

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

    def _build_encoder(self):
        ## encoder
        enc_inputs = self.x_ph
        gru_rnn1 = tf.nn.rnn_cell.GRUCell(32)
        gru_rnn2 = tf.nn.rnn_cell.GRUCell(16)
        enc_cell = tf.nn.rnn_cell.MultiRNNCell([gru_rnn1, gru_rnn2])
        _, enc_state = tf.nn.dynamic_rnn(
            enc_cell, enc_inputs,
            sequence_length=self.x_len_ph, dtype=tf.float32
        )
        return enc_state

    def _build_decoder(self, enc_state):
        ##todo: need to finish network with new api
        ## decoder
        dec_rnn1 = tf.nn.rnn_cell.GRUCell(32)
        dec_rnn2 = tf.nn.rnn_cell.GRUCell(16)
        dec_cell = tf.nn.rnn_cell.MultiRNNCell([dec_rnn1, dec_rnn2])

        #Dense layer to translate the decoder's output at each time
        fc_layer = tf.layers.Dense(self.out_dim, dtype=tf.float32)


        #Training Decoder
        with tf.variable_scope("predictor"):
            ## training decorder
            go_tokens = tf.constant(self.go_token, shape=[self.batch_size,1, self.out_dim])
            dec_input = tf.concat([go_tokens, self.y_ph], axis=1)

            seq_length = tf.constant(self.out_timesteps, shape=[self.batch_size])

            print("dec inputs shape:")
            print(dec_input.shape)

            training_helper = tf.contrib.seq2seq.TrainingHelper(dec_input, seq_length)

            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell, helper=training_helper,
                initial_state=enc_state, output_layer=fc_layer)

            training_decoder_outputs, training_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=self.out_timesteps
            )

            print("final_outputs shape:")
            print(training_decoder_outputs)

            # self.y_hat = training_decoder_outputs[0]

        #Inference Decoder
        with tf.variable_scope("predictor", reuse=True):
            ## inference decorder
            start_tokens = tf.constant(
                self.go_token, shape=[self.batch_size, self.out_dim])

            inference_helper = tf.contrib.seq2seq.InferenceHelper(
                sample_fn=lambda outputs: outputs,
                sample_shape=[self.out_dim],
                sample_dtype=tf.float32,
                start_inputs=start_tokens,
                end_fn=lambda sample_ids: False)

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell, helper=inference_helper,
                initial_state=enc_state, output_layer=fc_layer)

            inference_decoder_outputs, inference_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=self.out_timesteps
            )

            # self.y_hat = final_outputs[0]
        return training_decoder_outputs, inference_decoder_outputs

    def _build_net(self):

        ## encoder
        enc_state = self._build_encoder()

        ## docoder
        training_decoder_outputs, inference_decoder_outputs = self._build_decoder(enc_state)
        self.y_hat_train = training_decoder_outputs[0]
        self.y_hat_pred = inference_decoder_outputs[0]

        ## setup optimization
        self.loss = tf.losses.mean_squared_error(self.y_ph, self.y_hat_train)
        self.loss_pred = tf.losses.mean_squared_error(self.y_ph, self.y_hat_pred)

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

    def _create_seq(self, obs, dones, mean, var):
        """
        create sequences from input observations;
        reset sequence if a agent is done its task
        :param obs:  observations from environment
        :param dones: whether the agent is done its task
        :param mean: mean of observations
        :param var: variations of observations
        :return: done sequences
        """
        # todo: the format of this funtion need tobe edited
        if mean is not None and var is not None:
            ## save mean and var
            self.x_mean = np.concatenate((mean[6:13],
                                        mean[0:3])) #(joint angle, end-effector position)
            self.x_var = np.concatenate((var[6:13],
                                          var[0:3]))

        ## create sequence data
        for idx, data in enumerate(obs):
            self.xs[idx].append(np.concatenate((data[6:13],
                                                    data[0:3])))
            self.x_lens[idx] += 1


        ## reset requences that reaches destination
        seqs_done = []
        for idx, done in enumerate(dones):
            if done:
                # create a container saving reseted sequences for future usage
                seqs_done.append(DatasetStru(self.xs[idx], self.x_lens[idx],
                                            self.x_mean, self.x_var))
                self.xs[idx] = []
                self.x_lens[idx] = 0
                self.x_mean = np.zeros(self.in_dim)
                self.x_var = np.zeros(self.in_dim)
        return seqs_done

    def _create_dataset(self, seqs_done):
        """
        create dataset from saved sequences
        :return:
        """
        for data in seqs_done:
            if data.x_len > 1 and data.x_len < 300:
                self.dataset.append(data)
            # print("datasets size: {}".format(len(self.dataset)))

        # if dataset is large, save it
        if len(self.dataset) > 4000:
            print("save dataset...")
            pickle.dump(self.dataset, open("./model/"
                                           +"/dataset"+str(self.dataset_idx)+".pkl","wb"))
            self.dataset_idx+=1
            self.dataset=[]
            self.dataset=[]


    def _revert_data(self,data,mean,var):
        return(data*(var+1e-8)+mean)

    def _feed_training_data(self,dataset):
        xs, ys, x_lens = [], [], []

        for _ in range(0, self.batch_size):
            idx = random.randint(0, len(dataset) - 1)
            data = dataset[idx]
            length = data.x_len
            id = random.randint(1, length - 1)
            x, y, x_len = self._feed_one_data(data, id)
            xs.append(x)
            ys.append(y)
            x_lens.append(x_len)
        return xs, ys, x_lens

    def _feed_one_data(self,data,id):
        """
        #id: start index of this data
        #e.g.: x = data[id-self.in_timesteps:id];
        #      y = data[id:id+out_timesteps]
        :param data: a sequence data in DatasetStru format
        :return:
        """
        x = np.zeros((self.in_timesteps_max, self.in_dim))
        y = np.zeros((self.out_timesteps, self.out_dim))
        x_len = 0

        length = data.x_len
        id_start = id-self.in_timesteps_max
        id_end = id+self.out_timesteps

        if id_end>length:
            #pading dataset
            #todo: be careful, need to debug this function
            data.x = data.padding(data.x,id_end)

        if id_start>=0:
            x = data.x[id_start:id]
            y = data.x[id:id_end,-3:]
            x_len = self.in_timesteps_max
        elif id_start<0:
            x[0:id] = data.x[0:id]
            y = data.x[id:id_end,-3:]
            x_len = id


        # y_train = y.copy()
        # y_train = np.roll(y_train,1,axis=0)
        # y_train[0,:] = np.ones(self.out_dim)
        return x, y, x_len


    def _feed_online_data(self, data):
        xs, ys, x_lens = [], [], []
        for id in range(0,data.x_len):
            x, y, x_len = self._feed_one_data(data, id)
            xs.append(x)
            ys.append(y)
            x_lens.append(x_len)
        return xs, ys, x_lens


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
            xs, ys, x_lens=self._feed_training_data(self.dataset)
            #----start training-----#
            fetches = [self.train_op, self.merged_summary]
            fetches += [self.loss, self.y_ph, self.y_hat_train]
            feed_dict = {
                self.x_ph: xs,
                self.y_ph: ys,
                self.x_len_ph: x_lens
            }

            _, merged_summary, \
            loss, y, y_hat_train = self.sess.run(fetches, feed_dict)

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
                xs, ys, x_lens = self._feed_training_data(validate_set)

                ## run validation
                fetches = [self.loss_pred, self.x_ph, self.y_ph, self.y_hat_pred]
                feed_dict = {
                    self.x_ph: xs,
                    self.y_ph: ys,
                    self.x_len_ph: x_lens
                }
                loss_pred, x, y, y_hat_pred = self.sess.run(fetches, feed_dict)

                ## write summary
                validate_summary = tf.Summary()
                validate_summary.value.add(tag="validate rmse", simple_value=loss_pred)
                self.file_writer.add_summary(validate_summary, self.iteration)

                ## display
                # visualize.plot_3d_pred(x[0], y[0], y_hat[0])


                if (self.iteration % self.display_interval) is 0:
                    print('\n')
                    # print("x = {}".format(x[0]))
                    print("x_len={}".format(x_lens[0]))
                    print("pred = {}, true goal = {}".format(y[0], y_hat_pred[0]))
                    print('iteration = {}, validate loss = {} '.format(self.iteration, loss_pred))
            #---------create validate data-----#

        print("finish training")


    def collect(self, obs, dones, mean=None, var=None):
        """
        function: collect sequence dataset
        :param obs: obs.shape = [batch_size, ob_shape] include joint angle etc.
        :param dones: dones.shape = [batch_size]
        :param mean: mean.shape = [batch_size, ob_shape]
        :param var: var.shape = [batch_size, ob_shape]
        """

        #create input sequence
        seqs_done = self._create_seq(obs, dones, mean, var)

        #create training dataset for future training
        if len(seqs_done) > 0:
            if self.train_flag:
                self._create_dataset(seqs_done)


    def predict(self, obs, dones, mean=None, var=None):
        """
        Online predict sequence through trained model
        :param obs: obs.shape = [batch_size, ob_shape] include joint angle etc.
        :param dones: dones.shape = [batch_size]
        :param mean: mean.shape = [batch_size, ob_shape]
        :param var: var.shape = [batch_size, ob_shape]
        :return:
        """
        # create input sequence
        self._create_seq(obs, dones, mean, var)

        # ---predict input data---#
        # todo: rewrite this function, we need to feed data from a sequence
        for seq in seqs_done:
            xs, ys, x_lens = self._feed_online_data(seq)
            fetches = [self.loss, self.y_ph, self.y_hat]
            feed_dict = {
                self.x_ph: xs,
                self.y_ph: ys,
                self.x_len_ph: x_lens
            }

            loss, y, y_hat, y_train = self.sess.run(fetches, feed_dict)

            batch_loss = self._get_batch_loss(y, y_hat)

        ## display information
        if (self.iteration % self.display_interval) is 0:
            print('\n')
            print("pred = {}, true goal = {},y_train = {}".format(y_hat[0], y[0], y_train[0]))
            print('predict loss = {} '.format(loss))

        # #------plot predicted data-----------
        # import visualize
        # visualize.plot_3d_pred(xs[0],y[0],y_hat[0])
        # #------------------------------------#
        # print("batch loss")
        # print(batch_loss)


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
            rnn_model = Predictor(sess, FLAGS, 16, 10,
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

        else:
            #plot all the validate data step by step
            rnn_model = Predictor(sess, FLAGS, 16, 30,
                                  train_flag=False, reset_flag=False)

            rnn_model.init_sess()





