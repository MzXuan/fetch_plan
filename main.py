# -*- coding: utf-8 -*-

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
	initialize_env()
	#todo: get action from rl algorithm

	#todo: send observation to lstm

	return 0 




def initialize_env():
	env = gym.make('FetchPlan-v0')
	env.reset()



# env = gym.make('FetchPlan-v0')
# env.reset()

# for i in range(1000):
#     env.step(env.action_space.sample())
#     env.render()
#     time.sleep(0.1)



if __name__ == '__main__':
    main()
