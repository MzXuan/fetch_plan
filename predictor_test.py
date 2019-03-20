# -*- coding: utf-8 -*-
#general import

import numpy as np
import random

import tensorflow as tf
from flags import flags
from predictor import Predictor


FLAGS = flags.FLAGS

def main():
    train_flag=True


    with tf.Session() as sess:
        # create and initialize session
        rnn_model = Predictor(sess, FLAGS,32,10,train_flag=train_flag)
        rnn_model.initialize_sess()

        for i in range(0,5000):
            #create fake data
            obs = np.random.rand(32, 20)
            done = rand_bools_int_func(32)
            # run the model
            rnn_model.predict(obs,done)

    return 0

def rand_bools_int_func(n):
    r = random.getrandbits(n)
    return [bool((r>>i)&1) for i in range(n)]



if __name__ == '__main__':
    main()
