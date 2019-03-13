import gym
import gym_fetch
import time
env = gym.make('FetchPlan-v0')
# env = gym.make('HandReach-v0')
env.reset()

for i in range(1000):
    action= env.action_space.sample()
    # for idx,_ in enumerate(action):
    #     if idx ==5:
    #         action[idx]=0.8
    #     else:
    #         action[idx] = 0
    obs,_,done,_ = env.step(action)
    # env.reset()
    env.render()

    #todo: draw action, draw state
    joint_action = action
    joint_vel = obs['observation'][10:17]
    joint_pos = obs['observation'][3:10]
    print("joint action: ")
    print(joint_action)
    print("joint_pos: ")
    print(joint_pos)
    print("joint velocity: ")
    print(joint_vel)
    time.sleep(0.1)
