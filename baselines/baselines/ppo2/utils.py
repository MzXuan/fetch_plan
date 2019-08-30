import os, sys, glob, gc, joblib
import numpy as np
import math


def point_distance(point1,point2):
	return np.square(point1-point2).mean()


def find_goal(path, goals):
	'''
	find cloest goal to the path
	path: numpy array: timestep * dim
	goals: list; numpy goal
	'''
	dist_min = float("inf")

	for point in path:
		for idx, g in enumerate(goals):
			d = point_distance(point, g)
			if d < dist_min:
				dist_min = d
				goal_idx = idx

	# dis_list = []
	# for g in goals:
	# 	dis = np.linalg.norm((path - g), axis=1)
	# 	dis_list.append(dis.min())
	# print("min dist list: ", dis_list)

	return goals[goal_idx], goal_idx, dist_min


def GetRandomGoal(dims):
	goals = []
	limite = [-3, 3]
	max_min = limite[1] - limite[0]
	min = limite[0]
	for _ in range(5):
		goals.append(np.random.random(dims)*max_min+min)
	return goals


if __name__ == '__main__':
	GetRandomGoal(3)

	#    path = np.random.rand(10,3)
	# goals = []
	# for _ in range(5):
	# 	goals.append(np.random.rand(3))

	# goal_idx = find_goal(path, goals)