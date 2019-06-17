import os, sys, glob, gc, joblib
import numpy as np

def GetRandomGoal(dims):
    goals = []

    limite = [-1, 1]
    max_min = limite[1] - limite[0]
    min = limite[0]
    for _ in range(5):
        goals.append(np.random.random(dims)*max_min+min)
    return goals


if __name__ == '__main__':
    GetRandomGoal(3)