import copy

import numpy as np
import mujoco_py
from gym.envs.robotics import rotations, robot_env
from gym_fetch import utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchLSTMRewardEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold,  max_accel, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.maxi_accerl = max_accel
        self.maxi_vel = 0.5
        self.last_distance = 0.0

        self.current_qvel = np.zeros(7)
        self.prev_act = np.zeros(7)


        super(FetchLSTMRewardEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=7,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info, predict_reward=0):
        # predict reward: a predict reward from LSTM prediction algorithm
        # Compute distance between goal and the achieved goal.
        if info["is_success"]:
            return 30.0
        elif info["is_collision"]:
            return -20.0
        else:
            current_distance = goal_distance(achieved_goal, goal)
            approaching_rew = 15.0 * (self.last_distance - current_distance)
            self.last_distance = copy.deepcopy(current_distance)
            return approaching_rew

    def _reset_arm(self):
        collision_flag = True
        while collision_flag:
            initial_qpos = {
                'robot0:slide0': 0.4049,
                'robot0:slide1': 0.48,
                'robot0:slide2': 0.0,
                'robot0:torso_lift_joint': 0.0,
                'robot0:head_pan_joint': 0.0, #range="-1.57 1.57"
                'robot0:head_tilt_joint': 0.0, #range="-0.76 1.45"
                'robot0:shoulder_pan_joint': 2 * 1.6056 * np.random.random() - 1.6056, #range="-1.6056 1.6056"
                'robot0:shoulder_lift_joint': (1.221 + 1.518) * np.random.random() - 1.221, #range="-1.221 1.518"
                'robot0:upperarm_roll_joint': 2 * np.pi * np.random.random() - np.pi, #limited="false"
                'robot0:elbow_flex_joint': 2.251 * 2 * np.random.random() - 2.251, #range="-2.251 2.251"
                'robot0:forearm_roll_joint': 2 * np.pi * np.random.random() - np.pi, #limited="false"
                'robot0:wrist_flex_joint': 2.16 * 2 * np.random.random() - 2.16, #range="-2.16 2.16"
                'robot0:wrist_roll_joint': 2 * np.pi * np.random.random() - np.pi, #limited="false"
                'robot0:r_gripper_finger_joint': 0,
                'robot0:l_gripper_finger_joint': 0
            }

            for name, value in initial_qpos.items():
                self.sim.data.set_joint_qpos(name, value)
            self.current_qpos = self.sim.data.qpos[self.sim.model.jnt_qposadr[6:13]]
            self.initial_state = self.sim.get_state()
            self.sim.set_state(self.initial_state)
            self.sim.forward()
            collision_flag = self._contact_dection()

    # RobotEnv methods
    # ----------------------------
    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        self._reset_arm()
        self.prev_act = np.zeros(7)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        self.last_distance = goal_distance(
            obs['achieved_goal'], self.goal)
        return obs

    # RobotEnv methods
    # ----------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.sim.step()
        real_act = self._set_action(action)
        self._step_callback()
        obs = self._get_obs()
        done = False

        # self._contact_dection()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_collision': self._contact_dection()
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        # energy_loss = 0.2 * np.linalg.norm(real_act - self.prev_act)
        # # print("approching_rew: {} | energy_loss: {}".format(reward, energy_loss))
        # reward -= energy_loss
        self.prev_act = real_act.copy()
        if info["is_success"] or info["is_collision"]:
            done = True
        return obs, reward, done, info

    def _contact_dection(self):
        #----------------------------------
        # if there is collision: return true
        # if there is no collision: return false
        #----------------------------------

        # print('number of contacts', self.sim.data.ncon)
        # for i in range(self.sim.data.ncon):
        #     # Note that the contact array has more than `ncon` entries,
        #     # so be careful to only read the valid entries.
        #     contact = self.sim.data.contact[i]
        #     print('contact', i)
        #     print('dist', contact.dist)
        #     print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
        #     print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
        #     # There's more stuff in the data structure
        #     # See the mujoco documentation for more info!
        #     geom2_body = self.sim.model.geom_bodyid[self.sim.data.contact[i].geom2]
        #     print(' Contact force on geom2 body', self.sim.data.cfrc_ext[geom2_body])
        #     print('norm', np.sqrt(np.sum(np.square(self.sim.data.cfrc_ext[geom2_body]))))
        #     # Use internal functions to read out mj_contactForce
        #     c_array = np.zeros(6, dtype=np.float64)
        #     print('c_array', c_array)
        #     mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
        #     print('c_array', c_array)

        if self.sim.data.ncon > 10:
            return True
        else:
            return False

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):

        assert action.shape == (7,)
        action = action.copy()

        #-----------not use actuator, only self defined kinematics--------------------
        self.last_qvel = self.current_qvel
        self.last_qpos = self.current_qpos
        delta_v = np.clip(action-self.last_qvel, -self.maxi_accerl, self.maxi_accerl)
        action_clip = delta_v+self.last_qvel
        action_clip = np.clip(action_clip, -self.maxi_vel, self.maxi_vel)


        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        self.sim.data.qpos[self.sim.model.jnt_qposadr[6:13]] = self.last_qpos+(action_clip+1e-8)*dt

        self.current_qvel = action_clip
        self.current_qpos = self.sim.data.qpos[self.sim.model.jnt_qposadr[6:13]]

        return action_clip

        # #-----------directly control velocity------------------------
        # delta_v = np.clip(action - self.sim.data.qvel[6:13], -self.maxi_accerl, self.maxi_accerl)
        # action = delta_v + self.sim.data.qvel[6:13]
        # self.sim.data.qvel[6:13] = action

        #-----------directly control position------------------------

        # self.sim.data.qpos[self.sim.model.jnt_qposadr[6:13]] = action
        # print("bias: ")
        # print(self.sim.data.qfrc_bias)

        # #-------use actuator-----------
        # ctrlrange = self.sim.model.actuator_ctrlrange
        # # print("ctrlrange: ")
        # # print(ctrlrange)
        # action = np.expand_dims(action,axis=1)
        #
        # # Apply action to simulation.
        # utils.ctrl_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        # todo: add joint observation
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, _ = utils.robot_get_obs(self.sim)

        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)


        joint_angle = robot_qpos[6:13]
        joint_vel = self.current_qvel

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        
        obs = np.concatenate([
            joint_angle, joint_vel
        ])

        # obs = np.concatenate([
        #     joint_angle, joint_vel, self.prev_act
        # ])
        # ------------------------
        #   Observation details
        #   obs[0:3]: end-effector position
        #   obs[3:10]: joint angle
        #   obs[10:17]: joint velocity
        # ------------------------

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            # print("initial gripper xpos:")
            # print(self.initial_gripper_xpos)

            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 4.0:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        self.last_distance = 0
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = np.zeros(3)
            goal[0] = self.initial_gripper_xpos[0] + self.np_random.uniform(-0.20, 0.20, size=1)
            goal[1] = self.initial_gripper_xpos[1] + self.np_random.uniform(-0.40, 0.40, size=1)
            goal[2] = self.initial_gripper_xpos[2] + self.np_random.uniform(-0.4, 0.40, size=1)

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        # print("==> init_qpos:")
        for name, value in initial_qpos.items():
            # print("{} : {}".format(name, value))
            self.sim.data.set_joint_qpos(name, value)
        self.current_qpos = self.sim.data.qpos[self.sim.model.jnt_qposadr[6:13]]
        self.initial_state = self.sim.get_state()

        # print("done env initialization")

        self.sim.forward()
        self.sim.step()

        # for _ in range(10):
        #     self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchLSTMRewardEnv, self).render(mode, width, height)