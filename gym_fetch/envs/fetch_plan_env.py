import os
import numpy as np
from gym import utils
from gym_fetch.envs import fetch_LSTM_reward_env

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
            self, TEST_MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.1, max_accel=0.2,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _sample_goal(self):
        # todo: random choose a site object as the goal
        # check mujoco_py api about how to list all the site object

        goal = np.zeros(3)
        goal[0] = self.initial_gripper_xpos[0] + self.np_random.uniform(-0.20, 0.20, size=1)
        goal[1] = self.initial_gripper_xpos[1] + self.np_random.uniform(-0.40, 0.40, size=1)
        goal[2] = self.initial_gripper_xpos[2] + self.np_random.uniform(-0.4, 0.40, size=1)

        return goal.copy()

    def _render_callback(self):
        # Visualize a small ball on selected target
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()
