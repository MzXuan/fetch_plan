# -*- coding: utf-8 -*-
import time, os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import joblib
import pickle
import os
import time

import random
import numpy as np

import tensorflow as tf

# for plot saved dataset
import flags
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

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
        value = seq[-1, :]
        value = np.expand_dims(value, axis=0)
        for _ in range(old_length, new_length):
            seq = np.append(seq, value, axis=0)

        return seq


class FixedHelper(tf.contrib.seq2seq.InferenceHelper):
    """
    only required when batch size = 1
    """
    def sample(self, *args, **kwargs):
        result = super().sample(*args, **kwargs)
        # print("result size")
        # print(result)
        result.set_shape([1, 10]) #[batch_size, dimension]
        return result


class Predictor(object):
    def __init__(self, sess, FLAGS, 
                 batch_size, max_timestep, train_flag,
                 reset_flag=False, epoch=20, iter_start=0,
                 lr=0.001):
        ## extract FLAGS
        self.sess = sess
        self.start_iter = iter_start * epoch
        self.iteration = 0
        self.dataset_length = 0

        self._build_flag(FLAGS)

        self.batch_size = batch_size
        self.in_timesteps_max = max_timestep
        self.out_timesteps = 10
        self.train_flag = train_flag
        self.epochs = epoch
        self.lr = lr

        self.validate_ratio = 0.2

        ## prepare sequcne containers
        # self.xs = np.zeros((batch_size, self.in_timesteps_max, self.in_dim))
        self.xs = [[] for _ in range(0, self.batch_size)]
        self.x_lens = np.zeros(batch_size, dtype=int)
        self.x_mean = np.zeros(self.in_dim)
        self.x_var = np.zeros(self.in_dim)

        ## prepare containers for saving input dataset
        self.dataset = []
        if reset_flag:
            filelist = [f for f in os.listdir("./pred/") if f.endswith(".pkl")]
            # remove old files
            for f in filelist:
                os.remove(os.path.join("./pred/", f))

        self.dataset_idx = 0 # for counting the saved dataset index

        self.collect_flag = False
        ## build model
        self._build_ph()
        self._build_net()

    def _build_flag(self, FLAGS):        
        self.in_dim = FLAGS.in_dim
        self.out_dim = FLAGS.out_dim

        self.model_name = FLAGS.model_name

        self.validation_interval = FLAGS.validation_interval
        self.checkpoint_interval = FLAGS.checkpoint_interval
        self.sample_interval = FLAGS.sample_interval
        self.display_interval = FLAGS.display_interval
        self.checkpoint_dir = FLAGS.check_dir_cls
        self.sample_dir = FLAGS.sample_dir_cls

        
    def _build_ph(self):
        self.x_ph = tf.placeholder(
            tf.float32, 
            shape=[None, self.in_timesteps_max, self.in_dim],
            name='x_ph'
            )
            
        self.x_len_ph = tf.placeholder(
            tf.int32, 
            shape=[None], 
            name='in_timesteps_len'
            )

        self.y_train = tf.placeholder(
            tf.float32,
            shape=[None, self.out_timesteps, self.out_dim],
            name='y_train_ph'
            )

        self.y_ph = tf.placeholder(
            tf.float32, 
            shape=[None, self.out_timesteps, self.out_dim],
            name='y_ph'
            )

        self.decoder_seq_length = tf.placeholder(
            tf.int32,
            shape=[None],
            name='batch_seq_length'
            )

        self.go_token = np.full((self.out_dim), 0.0, dtype=np.float32)

    def init_sess(self):
        self.sess.run(tf.global_variables_initializer())

    def _build_encoder(self):
        ## encoder
        enc_inputs = self.x_ph
        gru_rnn1 = tf.nn.rnn_cell.GRUCell(32)
        gru_rnn2 = tf.nn.rnn_cell.GRUCell(32)
        enc_cell = tf.nn.rnn_cell.MultiRNNCell([gru_rnn1, gru_rnn2])

        _, enc_state = tf.nn.dynamic_rnn(
            enc_cell, enc_inputs,
            sequence_length=self.x_len_ph, dtype=tf.float32
        )
        return enc_state

    def _build_decoder(self, enc_state):
        ## decoder
        dec_rnn1 = tf.nn.rnn_cell.GRUCell(32)
        dec_rnn2 = tf.nn.rnn_cell.GRUCell(32)
        dec_cell = tf.nn.rnn_cell.MultiRNNCell([dec_rnn1, dec_rnn2])

        #Dense layer to translate the decoder's output at each time
        fc_layer = tf.layers.Dense(self.out_dim, dtype=tf.float32)

        #Training Decoder
        with tf.variable_scope("pred_dec"):
            ## training decorder
            go_tokens = tf.constant(self.go_token, shape=[self.batch_size, 1, self.out_dim])
            dec_input = tf.concat([go_tokens, self.y_ph], axis=1)

            seq_length = tf.constant(self.out_timesteps, shape=[self.batch_size])

            print("dec inputs shape:")
            print(dec_input.shape)

            training_helper = tf.contrib.seq2seq.TrainingHelper(dec_input, seq_length)

            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell, helper=training_helper,
                initial_state=enc_state, output_layer=fc_layer)

            training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=self.out_timesteps
            )

        #Inference Decoder
        with tf.variable_scope("pred_dec", reuse=True):
            ## inference decorder
            start_tokens = tf.constant(
                self.go_token, shape=[self.batch_size, self.out_dim])

            if self.batch_size == 1:
                inference_helper = FixedHelper(
                    sample_fn=lambda outputs: outputs,
                    sample_shape=[self.out_dim],
                    sample_dtype=tf.float32,
                    start_inputs=start_tokens,
                    end_fn=lambda sample_ids: False)
            else:
                inference_helper = tf.contrib.seq2seq.InferenceHelper(
                    sample_fn=lambda outputs: outputs,
                    sample_shape=[self.out_dim],
                    sample_dtype=tf.float32,
                    start_inputs=start_tokens,
                    end_fn=lambda sample_ids: False)

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell, helper=inference_helper,
                initial_state=enc_state, output_layer=fc_layer)

            inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=self.out_timesteps
            )
        return training_decoder_outputs, inference_decoder_outputs

    def _build_net(self):
        with tf.variable_scope("predictor"):
            ## encoder
            enc_state = self._build_encoder()
            ## docoder
            training_decoder_outputs, inference_decoder_outputs = self._build_decoder(enc_state)
            self.y_hat_train = training_decoder_outputs[0]
            self.y_hat_pred = inference_decoder_outputs[0]

            ## setup optimization
            self.training_loss = tf.losses.mean_squared_error(self.y_ph, self.y_hat_train)
            self.validate_loss = tf.losses.mean_squared_error(self.y_ph, self.y_hat_pred)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
            self.training_loss)

        ## save summary
        self.merged_summary = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(
            self.checkpoint_dir, self.sess.graph
            )

    def _get_batch_loss(self, ys, y_hats, x_lens):
        """
        calculate the mean square error between ground truth and prediction
        if ground truth y equals 0 (no ground truth), we set mean square error to 0
        :param ys: output
        :param y_hats: prediction
        :return: error
        """
        # #----- old version-----#
        # error = []
        # eff_weight = 0.7
        # for y, y_hat in zip(ys, y_hats):
        #     if not np.any(y[-1]):
        #         error.append(0)
        #     else:
        #         err1 = (1 - eff_weight) * \
        #                np.sum(np.square(y[:, 0:7] - y_hat[:, 0:7]))
        #         err2 = eff_weight * \
        #                np.sum(np.square(y[:, 7:10] - y_hat[:, 7:10]))
        #         error.append((np.sqrt(err1 + err2)))
        #
        # return np.asarray(error)

        #---- new normalized version ----#
        # err = err/delta(y)
        error = []
        eff_weight = 0.7
        for y, y_hat in zip(ys, y_hats):
            if not np.any(y[-1]):
                error.append(0)
            else:
                err1 = (1 - eff_weight) * \
                       np.sum(np.square(y[:, 0:7] - y_hat[:, 0:7])/
                              np.abs( np.cumsum(y[:, 0:7], axis=0)+1e-8))

                # print("y[:, 0:7]")
                # print(y[:, 0:7])
                # print("cumsum y")
                # print(np.cumsum(y[:, 0:7], axis=0))
                err2 = eff_weight * \
                       np.sum(np.square(y[:, 7:10] - y_hat[:, 7:10])/
                              np.abs(np.cumsum(y[:, 7:10], axis=0)+1e-8))

                # print("current error:")
                # print(err1+err2)
                error.append((np.sqrt(err1 + err2)))

        return np.asarray(error)


        
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
        if mean is not None and var is not None:
            ## save mean and var
            self.x_mean = np.concatenate((mean[6:13],
                                          mean[0:3])) 
            self.x_var = np.concatenate((var[6:13],
                                         var[0:3]))

        seqs_done, seqs_all = [], []
        for idx, (ob, done) in enumerate(zip(obs, dones)):
            if done:
                # create a container saving reseted sequences for future usage
                seqs_done.append(DatasetStru(self.xs[idx], self.x_lens[idx],
                                             self.x_mean, self.x_var))
                self.xs[idx] = []
                self.x_lens[idx] = 0

            self.xs[idx].append(np.concatenate((ob[6:13],
                                                ob[0:3])))
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
            if traj.x_len > self.in_timesteps_max and traj.x_len < 500:
                self.dataset.append(traj)
                self.dataset_length += (traj.x_len -\
                    (self.in_timesteps_max + self.out_timesteps))

        # if dataset is large, save it
        if self.dataset_length > 200000:
            print("save dataset...")
            pickle.dump(self.dataset,
                open("./pred/" + "/dataset_new" + ".pkl", "wb"))
            self.collect_flag = True

    def _revert_data(self, data, mean, var):
        return data * (var + 1e-8) + mean

    def _accumulate_data(self, delta_x, delta_y, x_start):
        x = delta_x + x_start
        y = delta_y + x_start
        return x, y

    def _process_dataset(self, trajs):
        xs, ys, x_lens, xs_start = [], [], [], []
        for traj in trajs:
            for i in range(self.in_timesteps_max,
                           traj.x_len - self.out_timesteps):
                x, y, x_len, x_start = self._feed_one_data(traj, i)
                xs.append(x)
                ys.append(y)
                x_lens.append(x_len)
                xs_start.append(x_start)

        xs=np.asarray(xs)
        ys=np.asarray(ys)
        x_lens=np.asarray(x_lens)
        xs_start=np.asarray(xs_start)
        return [xs, ys, x_lens, xs_start]

    def _feed_one_data(self, data, ind):
        """
        #ind: start index of this data
        #e.g.: x = data[id-self.in_timesteps:id] - data[id-self.in_timesteps];
        #      y = data[id:id+out_timesteps]
        :param data: a sequence data in DatasetStru format
        :return:
        """
        x = np.zeros((self.in_timesteps_max, self.in_dim))
        y = np.zeros((self.out_timesteps, self.out_dim))
        x_len = 0

        length = data.x_len
        ind_start = ind - self.in_timesteps_max
        ind_end = ind + self.out_timesteps

        if ind_end > length:
            #pading dataset
            x_seq = data.padding(data.x, ind_end)
        else:
            x_seq = data.x

        if ind_start > 0:
            x_origin = x_seq[ind_start:ind, :]
            x_start = x_seq[ind_start-1, :]
            x = x_origin - x_start
            y = x_seq[ind:ind_end, :] - x_start
            x_len = self.in_timesteps_max

        elif ind_start <= 0:
            x_origin = x_seq[0:ind, :]
            x_start = x_seq[0, :]
            x = np.full((self.in_timesteps_max, self.in_dim), 0.0)
            x[self.in_timesteps_max-ind:self.in_timesteps_max,:] = x_origin - x_start
            y = x_seq[ind:ind_end, :] - x_start
            x_len = ind

        return x, y, x_len, x_start

    def _feed_online_data(self, sequences):
        xs, ys, x_lens, xs_start = [], [], [], []
        for data in sequences:
            length = data.x_len
            # print("current data length")
            # print(data.x_len)
            if length <= self.in_timesteps_max:
                x, y, x_len, x_start = self._feed_one_data(data, length)
                y = np.zeros((self.out_timesteps, self.out_dim))

            elif length < self.in_timesteps_max + self.out_timesteps:
                x, y, x_len, x_start = self._feed_one_data(data, self.in_timesteps_max)
                y[-1, :] = np.zeros(self.out_dim)

            elif length < self.in_timesteps_max + self.out_timesteps + 2:
                ind = length - self.out_timesteps
                x, y, x_len, x_start = self._feed_one_data(data, ind)
                y[-1, :] = np.zeros(self.out_dim)

            else:
                ind = length - self.out_timesteps
                x, y, x_len, x_start = self._feed_one_data(data, ind)

            xs.append(x)
            ys.append(y)
            x_lens.append(x_len)
            xs_start.append(x_start)

        return xs, ys, x_lens, xs_start

    def run_training(self):
        ## check whether in training
        if not self.train_flag:
            print("Not in training process,return...")
            return 0

        ## load dataset
        self._load_train_set()
        print("dataset_length: ", len(self.dataset))
        valid_len = int(self.validate_ratio * len(self.dataset))
        print("validate_length: ", valid_len)
        train_set = self._process_dataset(self.dataset[0:-valid_len])
        valid_set = self._process_dataset(self.dataset[-valid_len:-1])
        ## run training

        dataset_length = train_set[0].shape[0]
        print("training_length: ", dataset_length)
        inds = np.arange(dataset_length)
        for e in tqdm(range(self.epochs)):
            np.random.shuffle(inds)
            total_loss = []
            for i in range(0, dataset_length, self.batch_size):
                start = i
                end = start + self.batch_size
                if end >= dataset_length:
                    end = dataset_length
                    start = dataset_length - self.batch_size

                mb_inds = inds[start:end]
                fetches = [self.train_op, 
                           self.training_loss]

                feed_dict = {
                    self.x_ph: train_set[0][mb_inds],
                    self.y_ph: train_set[1][mb_inds],
                    self.x_len_ph: train_set[2][mb_inds]
                }

                _, loss = self.sess.run(fetches, feed_dict)
                total_loss.append(loss)
            
            train_loss = np.mean(total_loss)

            ## validate
            validate_loss = self.validate(valid_set)
            ## save model
            self.save_net(("./pred/{}/{}").format(
                self.model_name, e
            ))
            ## add tensorboard
            summary = tf.Summary()
            summary.value.add(tag="train_loss", simple_value=train_loss)
            summary.value.add(tag="validate_loss", simple_value=validate_loss)
            self.file_writer.add_summary(summary, self.start_iter + e)
            print('epoch {}:  train: {} | validate: {}'.format(
                e + 1, train_loss, validate_loss))

    def validate(self, validate_set):
        ## run validation
        validate_length = validate_set[0].shape[0]
        inds = np.arange(validate_length)
        total_loss = []
        for i in range(0, validate_length, self.batch_size):
            start = i
            end = start + self.batch_size
            if end >= validate_length:
                break

            mb_inds = inds[start:end]
            feed_dict = {
                self.x_ph: validate_set[0][mb_inds],
                self.y_ph: validate_set[1][mb_inds],
                self.x_len_ph: validate_set[2][mb_inds]
            }
            loss = self.sess.run(self.validate_loss, feed_dict)
            total_loss.append(loss)

        validate_loss = np.mean(total_loss)
        return validate_loss

    def run_test(self):
        """
        run test on validation set
        """
        pass
        # Need to refacotr --------------------------------------------------

        # ## check saved data se
        # filelist = [f for f in os.listdir("./pred/") if f.endswith(".pkl")]
        # num_sets = len(filelist)-1
        # self.dataset_idx = 0

        # print("load validate dataset {}".format(filelist[-1]))
        # test_set = \
        #     pickle.load(open(os.path.join("./pred/", filelist[-1]), "rb"))
        # # ## prepare threshold to switch dataset
        # # max_iteration = int(self.point)
        # # iter_range = range(0,max_iteration,500)
        # # iter_idx=0
        # # print("iter_range")
        # # print(iter_range)
        # ## run training
        # for i in range(1,100):
        # #----------test process--------#
        #     ## create validate data
        #     xs, ys, x_lens, xs_start = self._feed_training_data(test_set)

        #     ## run validation
        #     fetches = [self.validate_loss, self.x_ph, self.y_ph, self.y_hat_pred]
        #     feed_dict = {
        #         self.x_ph: xs,
        #         self.y_ph: ys,
        #         self.x_len_ph: x_lens
        #     }
        #     loss_pred, x, y, y_hat_pred = self.sess.run(fetches, feed_dict)

        #     ## write summary
        #     summary = tf.Summary()
        #     summary.value.add(tag="validate_loss", simple_value=loss_pred)
        #     self.file_writer.add_summary(summary, self.iteration)

        #     # ------display information-----------#
        #     batch_loss = self._get_batch_loss(ys, y_hat_pred, x_lens)
        #     print("\nbatch_loss = {}".format(batch_loss))
        #     # print('\n')
        #     # print("x = {}".format(xs[0]))
        #     # print("x_len={}".format(x_lens[0]))
        #     # print("pred = {}, true goal = {}".format(y_hat_pred[0], y[0]))
        #     print('iteration = {}, validate loss = {} '.format(self.iteration, loss_pred))

        #     # ------plot predicted data-----------
        #     import visualize
        #     origin_x, origin_y = self._accumulate_data(xs[0], ys[0], xs_start[0])
        #     _, origin_y_pred = self._accumulate_data(xs[0], y_hat_pred[0], xs_start[0])
        #     visualize.plot_3d_seqs(origin_x, origin_y, origin_y_pred)
        #     time.sleep(2)

        # Need to refacotr --------------------------------------------------

    def collect(self, obs, dones, mean=None, var=None):
        """
        function: collect sequence dataset
        :param obs: obs.shape = [batch_size, ob_shape] include joint angle etc.
        :param dones: dones.shape = [batch_size]
        :param mean: mean.shape = [batch_size, ob_shape]
        :param var: var.shape = [batch_size, ob_shape]
        """

        #create input sequence
        seqs_done, _ = self._create_seq(obs, dones, mean, var)

        #create training dataset for future training
        if len(seqs_done) > 0:
            self._create_traj(seqs_done)

        # print("dataset length: ", self.dataset_length)
        return self.collect_flag

    def predict(self, obs, dones, mean=None, var=None):
        """
        Online predict sequence through trained model
        :param obs: obs.shape = [batch_size, ob_shape] include joint angle etc.
        :param dones: dones.shape = [batch_size]
        :param mean: mean.shape = [batch_size, ob_shape]
        :param var: var.shape = [batch_size, ob_shape]
        :return: batch_loss; batch_loss.shape = [batch_size]
        """
        # create input sequence
        _, seqs_all = self._create_seq(obs, dones, mean, var)

        # ---predict input data---#
        xs, ys, x_lens, _ = self._feed_online_data(seqs_all)

        fetches = [self.validate_loss, self.y_ph, self.y_hat_pred]
        feed_dict = {
            self.x_ph: xs,
            self.y_ph: ys,
            self.x_len_ph: x_lens
        }

        _, y, y_hat_pred = self.sess.run(fetches, feed_dict)

        batch_loss = self._get_batch_loss(y, y_hat_pred, x_lens)

        # ## display information
        # if (self.iteration % self.display_interval) == 0:
        #     print('\n')
        #     print("x = {}".format(xs[0]))
        #     print("pred = \n{}, true goal = \n{}".format(y_hat_pred[0], y[0]))
        #     print('predict loss = {} '.format(loss_pred))
        #     # print("batch_loss = {}".format(batch_loss))

        # # ------plot predicted data-----------
        # import visualize
        # origin_x, origin_y = self._accumulate_data(xs[0], ys[0], xs_start[0])
        # _, origin_y_pred = self._accumulate_data(xs[0], y_hat_pred[0], xs_start[0])
        # # visualize.plot_3d_seqs(origin_x, origin_y, origin_y_pred)
        # visualize.plot_dof_seqs(origin_x, origin_y, origin_y_pred)
        # #---------------------------

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

    def load(self):
        filename = ("./pred/" + self.model_name + "/{}").format(self.epochs-1)
        self.load_net(filename)

    def load_net(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="predictor"
        )

        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)

    # def save_dataset(self):
    #     # check whether in training process
    #     if self.train_flag is not True:
    #         print("Not in training process, saving failed")
    #     else:
    #         pickle.dump(self.dataset, open("./pred/"
    #                                         +"/dataset"+str(self.dataset_idx)+".pkl", "wb"))
    #         print("saving dataset successfully")
    #         self.dataset = []

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
        for idx in range(0, num_sets):
            dataset = self.load_dataset(filelist[idx])
            if dataset == 0:
                return 0
            else:
                self.dataset.extend(dataset)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--iter', default=0, type=int)
    args = parser.parse_args()

    train_flag=True
    FLAGS = flags.InitParameter()

    if not os.path.isdir("./pred"):
        os.mkdir("./pred")

    with tf.Session() as sess:
        if train_flag:
            # create and initialize session
            rnn_model = Predictor(sess, FLAGS, 1024, 10,
                                  train_flag=True, reset_flag=False, epoch=args.epoch,
                                  iter_start=args.iter, lr=args.lr)

            rnn_model.init_sess()

            if args.load:
                try:
                    rnn_model.load()
                    print("load model successfully")
                except:
                    rnn_model.init_sess()

            rnn_model.run_training()

        else:
            #plot all the validate data step by step
            rnn_model = Predictor(sess, FLAGS, 1, 10,
                                  train_flag=False, reset_flag=False, epoch=args.epoch)

            rnn_model.init_sess()
            rnn_model.load()
            rnn_model.run_test()