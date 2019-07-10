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

    fig3d = plt.figure(2)
    plt.clf()

    ax = fig3d.gca(projection='3d')
    plt.ion()


    ax.plot(obs_list[:, 0], obs_list[:, 1], obs_list[:, 2],
            '-o', linewidth=2, color="blue")

    plt.pause(0.1)

    return obs_list


def plot_3d_pred(x, goal, pred=None):

    fig3d = plt.figure(3)
    plt.clf()

    ax = fig3d.gca(projection='3d')
    plt.ion()

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    # print("obs_list[:,0]")
    # print(obs_list[:,0])
    ax.plot(x[:, -3], x[:, -2], x[:, -1],
            '-o', linewidth=2, color="blue", label="x")

    ax.plot([goal[0]], [goal[1]], [goal[2]],
            marker='o', markersize=10, color="brown", label="goal")
    if pred is not None:
        ax.plot([pred[0]], [pred[1]], [pred[2]],
                marker='o', markersize=10, color="red", label="pred")

    ax.legend()
    plt.pause(0.1)


def plot_3d_seqs(x, y_pred, y_true=None):
    fig3d = plt.figure(3)
    plt.clf()

    ax = fig3d.gca(projection='3d')
    plt.ion()

    # ax.set_xlim(-1.5, 1.5)
    # ax.set_ylim(-1.5, 1.5)
    # ax.set_zlim(-1.5, 1.5)

    ax.plot(x[:, -3], x[:, -2], x[:, -1],
            '-+', linewidth=2, color="blue", label="x")

    ax.plot(y_pred[:, -3], y_pred[:, -2], y_pred[:, -1],
            '-+', linewidth=2, color="red", label="pred", )

    ax.plot([y_pred[0, -3]], [y_pred[0, -2]], [y_pred[0, -1]],
            '*', color="red")

    ax.plot([y_pred[0, -3]], [y_pred[0, -2]], [y_pred[0, -1]],
            'o', color="red")


    if y_true is not None:
        ax.plot(y_true[:, -3], y_true[:, -2], y_true[:, -1],
                '-+', linewidth=2, color="green", label="y_true", alpha=0.5)


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.pause(0.1)



def plot_dof_seqs(x, y_pred, step = 1,  y_true=None, goals = None, goal_pred = None):
    plt.figure("dof")
    plt.ion()

    plt.clf()
    time_step_x = range(0,x.shape[0])
    time_step_y = range(x.shape[0], x.shape[0]+step*y_pred.shape[0], step)
    if y_true is not None:
        time_step_y_true = range(0, y_true.shape[0])

    DOFs = x.shape[-1]
    for j in range(0, DOFs):

        plt.subplot(DOFs,1,j+1)
        plt.ylim(-3, 3)
        plt.plot(time_step_x, x[:, j], "b-")
        plt.plot(time_step_y, y_pred[:, j], "r-", alpha = 0.6)

        if y_true is not None:
            plt.plot(time_step_y_true, y_true[:, j], color="green", alpha=0.5)
        if goals is not None:
            for goal in goals:
                plt.plot(time_step_y[-1], goal[j],'ro')
        if goal_pred is not None:
            plt.plot(time_step_y[-1], goal_pred[j], 'g*')

    plt.pause(0.1)

def plot_dist(dist_list):
    plt.figure("dist")
    plt.ion()
    plt.clf()

    plt.ylim(0,2)

    plt.title("distance")
    time_step= range(0,len(dist_list))
    plt.plot(time_step, dist_list, "b-*")

    plt.pause(0.1)



def plot_3d_eef(x):
    # colors = ['grey', 'brown','orange','olive','green','cyan',
    #           'blue','purple','pink','red','black','yellow']

    fig3d = plt.figure(3)
    ax = fig3d.gca(projection='3d')
    ax.plot(x[:, -3], x[:, -2], x[:, -1],
            '-+', linewidth=2, color='grey', label="x")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_avg_err(ratio, err_y, err_x):
    plt.figure("dist_err")
    plt.title("minimum distance to actually goal")
    plt.xlabel("observed percentage")
    plt.ylabel("distance / m^2")
    plt.plot(ratio, err_y,'r-', label='predict distance')
    plt.plot(ratio,err_x,'b-', label='observed distance')
    plt.legend()
    plt.show()
