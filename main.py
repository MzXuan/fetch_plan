# -*- coding: utf-8 -*-
#general import
import numpy as np
import random

# for gym env
import gym
import gym_fetch
import time

# for tensorflow
import tensorflow as tf
from flags import flags
from predictor import Predictor

FLAGS = flags.FLAGS


def main():
    print("import successfully")
    # env = initialize_env()

    # #-----------for debug----------------
    # obs = np.random.rand(32, 20)
    # for data in obs:
    #     test=np.concatenate((data[0:7],data[14:17]))
    #     print(test)
    #
    # #-------------end debug-------------

    with tf.Session() as sess:
        rnn_model = Predictor(sess, FLAGS,32,50,train_flag=True)
        rnn_model.initialize_sess()

        for i in range(0,50):
            obs = np.random.rand(32, 20)
            done = rand_bools_int_func(32)
            rnn_model.predict(obs,done)

    return 0

def rand_bools_int_func(n):
    r = random.getrandbits(n)
    return [bool((r>>i)&1) for i in range(n)]

def initialize_env():
    env = gym.make('FetchPlan-v0')
    env.reset()
    return env



# env = gym.make('FetchPlan-v0')
# env.reset()

# for i in range(1000):
#     env.step(env.action_space.sample())
#     env.render()
#     time.sleep(0.1)



if __name__ == '__main__':
    main()
