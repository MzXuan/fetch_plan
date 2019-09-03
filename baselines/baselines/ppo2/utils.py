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

	return goals[goal_idx], goal_idx, dist_min

def point_goal_reward(batched_seqs, x_starts, batched_goals, batch_alternative_goals):
	'''
	:param point:
	:param goals:
	:return:
	'''
	rewards = []
	n_envs = len(batched_seqs)
	for idx in range(0, n_envs):
		seq = batched_seqs[idx]
		total_length = len(seq)
		if total_length < 5:
			rewards.append(0.0)
		else:
			dis_list = []
			true_goal = batched_goals[idx] - x_starts[idx][-3:]
			alternative_goals = batch_alternative_goals[idx].reshape((3, 3)) - x_starts[idx][-3:]
			point = seq[-1,-3:]
			d0 = np.linalg.norm(point - true_goal)
			if d0 < 0.2:
				rew = 0.0
			else:
				for g in alternative_goals:
					if np.linalg.norm(g - true_goal) < 1e-7:
						pass
					else:
						dis_list.append(np.linalg.norm(point - g))
				rew = reward_dist(d0, dis_list, total_length)
			rewards.append(rew)
	return np.asarray(rewards)

def path_goal_reward(path, alternative_goals, g0, total_length):
	dis_list = []
	d0 = np.linalg.norm((path - g0), axis=1).min()
	for g in alternative_goals:
		if np.linalg.norm(g - g0) < 1e-7:
			pass
		else:
			dis = np.linalg.norm((path - g), axis=1).min()
			dis_list.append(dis)

	return(reward_dist(d0, dis_list,total_length))



def reward_dist(d0, dis, t):
	rew = []
	for d in dis:
		theta = 1 if d0 < d else -1
		# rew.append(theta*math.log(abs(d0-d)/abs(d0)+1))
		rew.append(theta*math.log(abs(d0-d)/abs(d0+d)+1))
	# print("distance is {} and reward list is: {} ".format(dist, rew))
	min_rew = np.asarray(rew).min()
	time_scale = math.exp(-t / 30)
	return (time_scale*min_rew)

# def goal_dist(point, gs,g0, t):
# 	rew = []
# 	dist = []
# 	d0 = np.linalg.norm(point - g0)
# 	if d0 < 0.2: # to close to give predictable reward
# 		return 0.0
# 	else:
# 		dist.append(d0)
# 		for g in gs:
# 			if np.linalg.norm(g - g0) < 1e-7:
# 				continue
# 			else:
# 				d = np.linalg.norm(point - g)
# 				theta = 1 if d0 < d else -1
# 				rew.append(theta*math.log(abs(d0-d)/abs(d0+d)+1))
# 				dist.append(d)
# 	# print("distance is {} and reward list is: {} ".format(dist, rew))
# 	min_rew = np.asarray(rew).min()
# 	time_scale = math.exp(-t / 30)
# 	return (time_scale*min_rew)

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