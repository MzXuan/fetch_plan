import gym
import gym_fetch
import time

import matplotlib.pyplot as plt



fig1 = plt.figure(1)
fig2 = plt.figure(2)
control,pos,vel = [],[],[]


env = gym.make('FetchPlan-v0')
# env = gym.make('HandReach-v0')
env.reset()

show_id = 1


for i in range(50):
    action= 0.5*env.action_space.sample()

    obs,_,done,_ = env.step(action)
    # env.reset()
    env.render()
    time.sleep(0.1)

    #todo: plot control variable and real variable
    joint_action = action
    joint_pos = obs['observation'][3:10]
    joint_vel = obs['observation'][10:17]

    # print("joint action: ")
    # print(joint_action)
    # print("joint_pos: ")
    # print(joint_pos)
    # print("joint velocity: ")
    # print(joint_vel)

    control.append(action[show_id])
    pos.append(joint_pos[show_id])
    vel.append(joint_vel[show_id])

    plt.figure(1)
    plt.ion()
    plt.cla()
    plt.plot(range(0,len(control)),control,'--',color="red")
    plt.plot(range(0, len(control)), vel, '-*', color="blue")

    plt.figure(2)
    plt.plot(range(0, len(control)), pos, '-*', color="blue" )
    plt.pause(0.1)

plt.ioff()
plt.show()




    # plt.cla()
    # plt.plot(range(0,len(control)),control,'--',color="red")
    # plt.plot(range(0, len(control)), state, '-*', color="blue")
    # plt.show()

