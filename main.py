# -*- coding: utf-8 -*-
#general import
import numpy as np

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
    env = initialize_env()
    with tf.Session() as sess:
        rnn_model = Predictor(sess, FLAGS)
        rnn_model.initialize_sess()

        x = np.empty([50,3])
        y = np.empty([1,3])
        x_len = 0
        for i in range(0,5000):

            #todo: get action from rl algorithm
            #todo: edit env that can return batch data
            obs, _, done, info = env.step(env.action_space.sample())
            # env.render()
            # time.sleep(0.1)
            # print("obs: ")
            # print(obs)
            # print("done: ")
            # print(done)
            desired_goal = obs['desired_goal']
            current_state = obs['observation'][0:3]
            # print("current state: ")
            # print(current_state)
            y = desired_goal
            if(x_len<10):
                x[x_len] = current_state
                x_len+=1
            else:
                x = np.roll(x,-1,axis=0)
                x[x_len-1] = current_state

            # print("x_len: ")
            # print(x_len)
            # print("x: ")
            # print(x)
            # print("y: ")
            # print(y)
            #todo: send observation to lstm 
            loss = rnn_model.train_online(x,y,x_len,i)

    return 0 


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
