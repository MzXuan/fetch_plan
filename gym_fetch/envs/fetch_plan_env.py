import os
import numpy as np
from gym import utils
from gym_fetch.envs import fetch_LSTM_reward_env

import random

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch','jointvel.xml')
TEST_MODEL_XML_PATH = os.path.join('fetch', 'jointvel_test.xml')
# print(MODEL_XML_PATH)


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
            obj_range=0.15, target_range=0.15, distance_threshold=0.05, max_accel=0.2,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


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
            initial_qpos=initial_qpos, reward_type=reward_type)
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

