import gym
import gym_fetch
import time
env = gym.make('FetchPlan-v0')
env.reset()

for i in range(1000):
    env.step(env.action_space.sample())
    env.render()
    time.sleep(0.1)