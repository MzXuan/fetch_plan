# -*- coding: utf-8 -*
import os
from datetime import datetime
import tensorflow as tf

num_units = 16
num_layers = 2
out_steps = 8
in_timesteps_max = 10

# model_name = "rl_{}_{}_{}_seq_tanh".format(num_units, num_layers, out_steps)
model_name = "seq2seq"

in_dim = 10
out_dim = 10

learning_rate = 1e-3

validation_interval = 50
checkpoint_interval = 500
sample_interval = 50
display_interval = 100

# check_dir_cls = './pred/' + model_name + '/checkpoint_'
# sample_dir_cls = './pred/' + model_name + '/sample_'

random_seed = 100



# class InitParameter():
#     def __init__(self, model_name="test1"):
#         self.model_name = model_name
#
#         model_name = "rl_{}_{}_{}_seq_tanh".format(NUM_UNITS, NUM_LAYERS, out_steps)
#
#         self.in_dim = 10
#         self.out_dim = 10
#         self.in_timesteps_max = 100
#         self.learning_rate = 1e-3
#
#         self.validation_interval = 50
#         self.checkpoint_interval = 500
#         self.sample_interval = 50
#         self.display_interval = 100
#
#         self.check_dir_cls = './pred/' + model_name + '/checkpoint_'
#         self.sample_dir_cls = './pred/' + model_name + '/sample_'

