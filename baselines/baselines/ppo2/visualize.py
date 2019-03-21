# _*_ coding: utf-8 _*_

"""
python_visual_animation.py by xianhu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D


def plot_obs(obs, obs_list=None):
    if obs_list is None:
        obs_list = obs[0, :3]
        obs_list = np.expand_dims(obs_list, axis=0)
    else:
        obs_list = np.vstack((obs_list, obs[0, :3]))
    
    # print("obs:")
    # print(obs)
    # print("obs_list")
    # print(obs_list)

    plt.figure(1)
    plt.ion()

    plt.clf()
    x = range(0, obs_list.shape[0])
    plt.plot(x, obs_list[:, 0], color="red")
    plt.plot(x, obs_list[:, 1], color="blue")
    plt.plot(x, obs_list[:, 2], color="yellow")
    plt.pause(0.1)
    return obs_list

def plot_3d_obs(obs, obs_list=None):
    if obs_list is None:
        obs_list = obs[0, :3]
        obs_list = np.expand_dims(obs_list, axis=0)
    else:
        obs_list = np.vstack((obs_list, obs[0, :3]))
    
    # print("obs:")
    # print(obs)
    # print("obs_list")
    # print(obs_list)

    fig3d =plt.figure(2)
    plt.clf()

    ax = fig3d.gca(projection='3d')
    plt.ion()

    # print("obs_list[:,0]")
    # print(obs_list[:,0])
    ax.plot(obs_list[:, 0], obs_list[:, 1], obs_list[:, 2], 
            '-o', linewidth=2, color="blue")
    
    plt.pause(0.1)

    return obs_list





