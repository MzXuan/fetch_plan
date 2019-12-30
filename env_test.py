import time
import gym
import gym_fetch
import numpy as np
import time
from datetime import datetime
import socket

port = 8001                             # 端口和上面一致
host = "localhost"                      # 服务器IP，这里服务器和客户端IP同一个
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# env = gym.make('FetchPlanTest-v0')
# env = gym.make('FetchPlan-v0')
env = gym.make('FetchPlanBoard-v0')
# env = gym.make('FetchPlanShelf-v0')
# env = gym.make('FetchEff-v0')
for e in range(100):
    env.reset()
    done = False
    total_rew = 0
    while not done:
        action = 0.6*env.action_space.sample()
        # print("action: ", action)
        obs, rew, done, info = env.step(action)
        env.render()
        # time.sleep(0.1)
        # print("rew: ", rew)
        eef_pos = obs['achieved_goal']
        delta_dist = obs['observation'][-3:]
        alter_goals = info['alternative_goals']
        end_flag = np.array([1]) if info['is_success'] or info['is_collision'] else np.array([0])

        now = datetime.now()
        msg_time = str(now.minute)+str(now.second)+"."+str(now.microsecond)
        # data_send = np.concatenate([eef_pos,delta_dist,alter_goals])
        
        data_send = np.array2string(np.concatenate([eef_pos,delta_dist,alter_goals,end_flag]), precision=3, separator=',', suppress_small=True)

        # print("data_send", data_send)
        sock.sendto((data_send+"t:"+str(msg_time)).encode(),(host, port))

        total_rew += rew
    print("episode {} rewards: {} ".format(
        e, total_rew
    ))



