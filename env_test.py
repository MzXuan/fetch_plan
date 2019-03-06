import gym
import time
env = gym.make('FetchReach-v1')
env.reset()

for i in range(1000):
    env.step(env.action_space.sample())
    env.render()
    time.sleep(0.1)