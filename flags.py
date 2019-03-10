# -*- coding: utf-8 -*
import os
from datetime import datetime
import tensorflow as tf


flags = tf.app.flags

##model name
flags.DEFINE_string('model_name','human_predict_test','name of the model')


## model hyper-parameters
flags.DEFINE_integer('num_units_cls', 32, 'number of units of a rnn cell in encoder or decoder')
flags.DEFINE_integer('num_units', 32, 'number of units of a rnn cell in encoder or decoder')
flags.DEFINE_integer('num_stacks', 1, 'number of stacked rnn cells in encoder or decoder')

flags.DEFINE_integer('in_dim', 3, 'dimensionality of each timestep input')
flags.DEFINE_integer('out_dim', 3, 'dimensionality of each timestep output')
flags.DEFINE_integer('num_cls', 7, 'task numbers')
# add weights for each dimensionality of output
flags.DEFINE_integer("loss_mode", 0, "0: calculate loss as a whole; 1: calculate loss one by one")
flags.DEFINE_integer('out_dim_wgt1', 1, 'The 1th weight for each dimensionality of output')
flags.DEFINE_integer('out_dim_wgt2', 1, 'The 2th weight for each dimensionality of output')
flags.DEFINE_integer('out_dim_wgt3', 1, 'The 3th weight for each dimensionality of output')
#
flags.DEFINE_integer('in_timesteps', 20, 'input timesteps')
flags.DEFINE_integer('in_timesteps_min', 10, 'input min timesteps')
flags.DEFINE_integer('in_timesteps_max', 50, 'input max timesteps')
flags.DEFINE_integer('out_timesteps', 50, 'output timesteps')

## optimization hyper-parameters
flags.DEFINE_integer('cls_max_iteration', 30000, 'max iteration of classification model')
flags.DEFINE_integer('max_iteration', 30000, 'max iteration of training model')
flags.DEFINE_integer('batch_size', 32, 'batch size of datapoints')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate of optimization')

## todo & can be adjusted
flags.DEFINE_integer("run_mode", 0, "0: training; 1: testing; 2: test by trajectory; 3: testing online")
flags.DEFINE_integer("test_type", 0, "0: testing as a whole; 1: testing one by one")

## log hyper-parameters
flags.DEFINE_integer('validation_interval', 500, 'interval of performing validation')
flags.DEFINE_integer('checkpoint_interval', 500, 'interval of saving checkpoint')
flags.DEFINE_integer('sample_interval', 500, 'interval of sampling datapoints')
flags.DEFINE_integer('display_interval', 100, 'interval of displaying information')

## log directory 
stamp = 'stamp' + datetime.now().strftime("%Y%m%d-%H%M-%S")
num_units = 'units' + str(flags.FLAGS.num_units)
num_stacks = 'stacks' + str(flags.FLAGS.num_stacks)
in_dim = 'indim' + str(flags.FLAGS.in_dim)
out_dim = 'outdim' + str(flags.FLAGS.out_dim)
in_timesteps = 'intime' + str(flags.FLAGS.in_timesteps)
out_timesteps = 'outtime' + str(flags.FLAGS.out_timesteps)
batch_size = 'batch' + str(flags.FLAGS.batch_size)

postfix = "ours_gru_64_64_output_15_w1_1_w2_10"

model_name = flags.FLAGS.model_name

# save directory of classifier
check_dir_cls = './model/'+model_name+'/cls'+'/checkpoint_' + postfix
sample_dir_cls = './model/'+model_name+'/cls'+'/sample_' + postfix

flags.DEFINE_string('check_dir_cls', check_dir_cls, 'Directory name to save checkpoints')
flags.DEFINE_string('sample_dir_cls', sample_dir_cls, 'Directory name to save datapoints')


# save directory of generator
checkpoint_dir = './model/'+model_name+'/gen'+'/checkpoint_' + postfix
sample_dir = './model/'+model_name+'/gen'+'/sample_' + postfix

flags.DEFINE_string('checkpoint_dir', checkpoint_dir, 'Directory name to save class checkpoints')
flags.DEFINE_string('sample_dir', sample_dir, 'Directory name to save class datapoints')


FLAGS = flags.FLAGS
