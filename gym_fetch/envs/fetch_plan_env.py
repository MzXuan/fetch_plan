import os
import numpy as np
from gym import utils
from gym_fetch.envs import fetch_LSTM_reward_env
from gym.envs.robotics import utils as utils_rob


import random

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch','jointvel.xml')
EFF_MODEL_XML_PATH = os.path.join('fetch', 'eff_point.xml')
TEST_MODEL_XML_PATH = os.path.join('fetch', 'jointvel_test.xml')

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)



class FetchPlanEnv(fetch_LSTM_reward_env.FetchLSTMRewardEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'robot0:torso_lift_joint': 0.04,
            'robot0:head_pan_joint': 0.0, #range="-1.57 1.57"
            'robot0:head_tilt_joint': 0.00, #range="-0.76 1.45"
            'robot0:shoulder_pan_joint': 0, #range="-1.6056 1.6056"
            'robot0:shoulder_lift_joint': 1.0, #range="-1.221 1.518"
            'robot0:upperarm_roll_joint': 0, #limited="false"
            'robot0:elbow_flex_joint': -2.0, #range="-2.251 2.251"
            'robot0:forearm_roll_joint': 0,#limited="false"
            'robot0:wrist_flex_joint':0.8, #range="-2.16 2.16"
            'robot0:wrist_roll_joint': 0, #limited="false"
            'robot0:r_gripper_finger_joint': 0,
            'robot0:l_gripper_finger_joint': 0
        }
        fetch_LSTM_reward_env.FetchLSTMRewardEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.07, max_accel=0.2,
            initial_qpos=initial_qpos, reward_type=reward_type, n_actions=7)
        utils.EzPickle.__init__(self)

    def _sample_goal(self):
        #get all the targets ids; random sample goal on a plane
        body_num = self.sim.model.body_name2id('target_plane')
        site_body_list = self.sim.model.site_bodyid
        index = np.where(site_body_list==body_num)[0] #1~number


        # random select one as the goal
        id = np.random.choice(a=index,size=1)

        goal = self.sim.data.site_xpos[id].reshape(3,)

        # print("origin goal: ", goal)

        # random goal y and z
        goal[1] = np.random.uniform()*2*0.6*goal[1]+(1-0.6)*goal[1]
        goal[2] = np.random.uniform()*2*0.6*goal[2]+(1-0.6)*goal[2]

        # print("randomed goal: ", goal)

        return goal.copy()

class FetchEffEnv(fetch_LSTM_reward_env.FetchLSTMRewardEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'robot0:torso_lift_joint': 0.04,
            'robot0:head_pan_joint': 0.0, #range="-1.57 1.57"
            'robot0:head_tilt_joint': 0.00, #range="-0.76 1.45"
            'robot0:shoulder_pan_joint': 0, #range="-1.6056 1.6056"
            'robot0:shoulder_lift_joint': 1.0, #range="-1.221 1.518"
            'robot0:upperarm_roll_joint': 0, #limited="false"
            'robot0:elbow_flex_joint': -2.0, #range="-2.251 2.251"
            'robot0:forearm_roll_joint': 0,#limited="false"
            'robot0:wrist_flex_joint':0.8, #range="-2.16 2.16"
            'robot0:wrist_roll_joint': 0, #limited="false"
            'robot0:r_gripper_finger_joint': 0,
            'robot0:l_gripper_finger_joint': 0
        }
        fetch_LSTM_reward_env.FetchLSTMRewardEnv.__init__(
            self, EFF_MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05, max_accel=0.2,
            initial_qpos=initial_qpos, reward_type=reward_type, n_actions=4)
        utils.EzPickle.__init__(self)


    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 1., 0., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils_rob.ctrl_set_action(self.sim, action)
        utils_rob.mocap_set_action(self.sim, action)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_collision': self._contact_dection(),
            'goal_label': self.goal_label
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        if info["is_success"] or info["is_collision"]:
            done = True


        return obs, reward, done, info


    def _sample_goal(self):
        #get all the targets ids; random sample goal on a plane
        body_num = self.sim.model.body_name2id('target_plane')
        site_body_list = self.sim.model.site_bodyid
        index = np.where(site_body_list==body_num)[0] #1~number
        # random select one as the goal
        id = np.random.choice(a=index,size=1)
        goal = self.sim.data.site_xpos[id].reshape(3,)

        # random goal y and z
        goal[1] = np.random.uniform()*2*0.6*goal[1]+(1-0.6)*goal[1]
        goal[2] = np.random.uniform()*2*0.6*goal[2]+(1-0.6)*goal[2]

        return goal.copy()

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        self.last_distance = goal_distance(
            obs['achieved_goal'], self.goal)
        return obs

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils_rob.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]



class FetchPlanTestEnv(fetch_LSTM_reward_env.FetchLSTMRewardEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'robot0:torso_lift_joint': 0.04,
            'robot0:head_pan_joint': 0.0, #range="-1.57 1.57"
            'robot0:head_tilt_joint': 0.00, #range="-0.76 1.45"
            'robot0:shoulder_pan_joint': 0, #range="-1.6056 1.6056"
            'robot0:shoulder_lift_joint': 0.8, #range="-1.221 1.518"
            'robot0:upperarm_roll_joint': 0, #limited="false"
            'robot0:elbow_flex_joint': -2.0, #range="-2.251 2.251"
            'robot0:forearm_roll_joint': 0,#limited="false"
            'robot0:wrist_flex_joint':0.8, #range="-2.16 2.16"
            'robot0:wrist_roll_joint': 0, #limited="false"
            'robot0:r_gripper_finger_joint': 0,
            'robot0:l_gripper_finger_joint': 0
        }
        fetch_LSTM_reward_env.FetchLSTMRewardEnv.__init__(
            self, TEST_MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.06, max_accel=0.2,
            initial_qpos=initial_qpos, reward_type=reward_type, n_actions=7)
        utils.EzPickle.__init__(self)


    def _sample_goal(self):
        '''random select pre-defined target as the final target
        '''
        #get all the targets ids
        body_num = self.sim.model.body_name2id('targets')
        site_body_list = self.sim.model.site_bodyid
        index = np.where(site_body_list==body_num)[0] #1~number

        # random select one as the goal
        id = np.random.choice(a=index,size=1)


        # id = np.random.choice([1, 12])
        # id = 1
        self.goal_label = int(id)-1
        # print("sampled id is: {}".format(id))
        goal = self.sim.data.site_xpos[id].reshape(3,)

        return goal.copy()


    def _render_callback(self):
        # Visualize a small ball on selected target
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_arm(self):
        collision_flag = True
        d = 0.1
        while collision_flag:
            # # random
            # initial_qpos = {
            #     'robot0:slide0': 0.4049,
            #     'robot0:slide1': 0.48,
            #     'robot0:slide2': 0.0,
            #     'robot0:torso_lift_joint': 0.0,
            #     'robot0:head_pan_joint': 0.0,  # range="-1.57 1.57"
            #     'robot0:head_tilt_joint': 0.0,  # range="-0.76 1.45"
            #     'robot0:shoulder_pan_joint': 2 * 1.6056 * np.random.random() - 1.6056,  # range="-1.6056 1.6056"
            #     'robot0:shoulder_lift_joint': (1.221 + 1.518) * np.random.random() - 1.221,  # range="-1.221 1.518"
            #     'robot0:upperarm_roll_joint': 2 * np.pi * np.random.random() - np.pi,  # limited="false"
            #     'robot0:elbow_flex_joint': 2.251 * 2 * np.random.random() - 2.251,  # range="-2.251 2.251"
            #     'robot0:forearm_roll_joint': 2 * np.pi * np.random.random() - np.pi,  # limited="false"
            #     'robot0:wrist_flex_joint': 2.16 * 2 * np.random.random() - 2.16,  # range="-2.16 2.16"
            #     'robot0:wrist_roll_joint': 2 * np.pi * np.random.random() - np.pi,  # limited="false"
            #     'robot0:r_gripper_finger_joint': 0,
            #     'robot0:l_gripper_finger_joint': 0
            # }
            #middle + random
            # initial_qpos = {
            #     'robot0:slide0': 0.4049,
            #     'robot0:slide1': 0.48,
            #     'robot0:slide2': 0.0,
            #     'robot0:torso_lift_joint': 0.0,
            #     'robot0:head_pan_joint': 0.0,  # range="-1.57 1.57"
            #     'robot0:head_tilt_joint': 0.0,  # range="-0.76 1.45"
            #     'robot0:shoulder_pan_joint': 0.0,  # range="-1.6056 1.6056"
            #     'robot0:shoulder_lift_joint': 0.8+d-2*d*np.random.random(), #range="-1.221 1.518"
            #     'robot0:upperarm_roll_joint': 0, #limited="false"
            #     'robot0:elbow_flex_joint': -1.9+d-2*d*np.random.random(), #range="-2.251 2.251"
            #     'robot0:forearm_roll_joint': 0,#limited="false"
            #     'robot0:wrist_flex_joint':1.5+d-2*d*np.random.random(), #range="-2.16 2.16"
            #     'robot0:wrist_roll_joint': 0, #limited="false"
            #     'robot0:r_gripper_finger_joint': 0,
            #     'robot0:l_gripper_finger_joint': 0
            # }

            #middle
            initial_qpos = {
                'robot0:slide0': 0.4049,
                'robot0:slide1': 0.48,
                'robot0:slide2': 0.0,
                'robot0:torso_lift_joint': 0.0,
                'robot0:head_pan_joint': 0.0,  # range="-1.57 1.57"
                'robot0:head_tilt_joint': 0.0,  # range="-0.76 1.45"
                'robot0:shoulder_pan_joint': 0.0,  # range="-1.6056 1.6056"
                'robot0:shoulder_lift_joint': 0.8, #range="-1.221 1.518"
                'robot0:upperarm_roll_joint': 0, #limited="false"
                'robot0:elbow_flex_joint': -1.9, #range="-2.251 2.251"
                'robot0:forearm_roll_joint': 0,#limited="false"
                'robot0:wrist_flex_joint':1.5, #range="-2.16 2.16"
                'robot0:wrist_roll_joint': 0, #limited="false"
                'robot0:r_gripper_finger_joint': 0,
                'robot0:l_gripper_finger_joint': 0
            }

            # # left
            # initial_qpos = {
            #     'robot0:slide0': 0.4049,
            #     'robot0:slide1': 0.48,
            #     'robot0:slide2': 0.0,
            #     'robot0:torso_lift_joint': 0.0,
            #     'robot0:head_pan_joint': 0.0,  # range="-1.57 1.57"
            #     'robot0:head_tilt_joint': 0.0,  # range="-0.76 1.45"
            #     'robot0:shoulder_pan_joint':0.5,  # range="-1.6056 1.6056"
            #     'robot0:shoulder_lift_joint': 0,  # range="-1.221 1.518"
            #     'robot0:upperarm_roll_joint': -1.0,  # limited="false"
            #     'robot0:elbow_flex_joint': 1.5,  # range="-2.251 2.251"
            #     'robot0:forearm_roll_joint': 0,  # limited="false"
            #     'robot0:wrist_flex_joint':1.0,  # range="-2.16 2.16"
            #     'robot0:wrist_roll_joint': 0,  # limited="false"
            #     'robot0:r_gripper_finger_joint': 0,
            #     'robot0:l_gripper_finger_joint': 0
            # }

            for name, value in initial_qpos.items():
                self.sim.data.set_joint_qpos(name, value)
            self.current_qpos = self.sim.data.qpos[self.sim.model.jnt_qposadr[6:13]]
            self.initial_state = self.sim.get_state()
            self.sim.set_state(self.initial_state)
            self.sim.forward()
            collision_flag = self._contact_dection()

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:torso_lift_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 3
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -14.

