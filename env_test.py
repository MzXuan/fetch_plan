import gym
import gym_fetch

env = gym.make('FetchPlan-v0')
for e in range(100):
    env.reset()
    done = False
    total_rew = 0
    while not done:
        action= 0.6*env.action_space.sample()
        obs, rew, done,_ = env.step(action)
        env.render()
        total_rew += rew
    print("episode {} rewards: {} ".format(
        e, total_rew
    ))



