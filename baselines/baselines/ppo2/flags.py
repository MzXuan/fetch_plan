# -*- coding: utf-8 -*
import os
from datetime import datetime
import tensorflow as tf

def InitParameter(model_name = "test1"):
    flags = tf.app.flags

    ##model name
    flags.DEFINE_string('model_name',model_name,'name of the model')

    ## model hyper-parameters
    flags.DEFINE_integer('in_dim', 7, 'dimensionality of each timestep input')
    flags.DEFINE_integer('out_dim', 7, 'dimensionality of each timestep output')

    flags.DEFINE_integer('in_timesteps_max', 50, 'input max timesteps')

    ## optimization hyper-parameters
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate of optimization')

    ## todo & can be adjusted
    flags.DEFINE_integer("run_mode", 0, "0: training; 1: testing; 2: test by trajectory; 3: testing online")

    ## log hyper-parameters
    flags.DEFINE_integer('validation_interval', 50, 'interval of performing validation')
    flags.DEFINE_integer('checkpoint_interval', 500, 'interval of saving checkpoint')
    flags.DEFINE_integer('sample_interval', 500, 'interval of sampling datapoints')
    flags.DEFINE_integer('display_interval', 100, 'interval of displaying information')


    # model_name = flags.FLAGS.model_name

    # save directory of classifier
    check_dir_cls = './model/'+model_name+'/checkpoint_'
    sample_dir_cls = './model/'+model_name+'/sample_'

    flags.DEFINE_string('check_dir_cls', check_dir_cls, 'Directory name to save checkpoints')
    flags.DEFINE_string('sample_dir_cls', sample_dir_cls, 'Directory name to save datapoints')

    return flags.FLAGS
