# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

def draw_trajs(preds, true):
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
    # print robot trajectory
    for i, sample in enumerate(preds):
        # sample = sample.T
        # print (sample[:,0])
        if i ==0:
            ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], '-o', linewidth=2, color='black', label = "predict mean")
        else:
            ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], '-o', linewidth=2, color='blue', label="predict stds")

    ax.plot(true[:, 0], true[:, 1], true[:, 2], '-o', linewidth=2, color='red')
    plt.show()
