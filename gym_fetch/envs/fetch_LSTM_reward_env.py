import numpy as np
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


        self.current_qvel = np.zeros(7)


        super(FetchLSTMRewardEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=7,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info, predict_reward=0):
        # todo: get reward from outside
        # predict reward: a predict reward from LSTM prediction algorithm
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)+predict_reward
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d


    # RobotEnv methods
    # ----------------------------
    def step(self, action):
        # print("self.action_space.low")
        # print(self.action_space.low)
        # print("self.action_space.high")
        # print(self.action_space.high)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info


    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()


    def _set_action(self, action):
        #todo: add a limitation of maximum accerleration and joint position(in xml file)
        # adjust pid

        assert action.shape == (7,)
        action = action.copy()

        #-----------not use actuator, only self defined kinematics--------------------
        self.last_qvel = self.current_qvel
        self.last_qpos = self.current_qpos
        delta_v = np.clip(action-self.last_qvel, -self.maxi_accerl, self.maxi_accerl)
        action_clip = delta_v+self.last_qvel

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        self.sim.data.qpos[self.sim.model.jnt_qposadr[6:13]] = self.last_qpos+action_clip*dt

        self.current_qvel = action_clip
        self.current_qpos = self.sim.data.qpos[self.sim.model.jnt_qposadr[6:13]]


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
        return 0




    def _get_obs(self):
        # positions
        # todo: add joint observation
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, _ = utils.robot_get_obs(self.sim)
        # print("robot qpos: ")
        # print(robot_qpos)
        # print("robot qvel: ")
        # print(robot_qvel)

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
            grip_pos, joint_angle, joint_vel
        ])
        # ------------------------
        #   Observation details
        #   obs[0:3]: end-effector position
        #   obs[3:10]: joint angle
        #   obs[10:17]: joint velocity
        # ------------------------

        # obs = np.concatenate([
        #     grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
        #     object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        # ])
        # print("grip_pos:")
        # print(grip_pos)
        # print("object_pos:")
        # print(object_pos)
        # print("obs:")
        # print(obs)
        # print("goal:")
        # print(self.goal)

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
        self.sim.set_state(self.initial_state)
        print('self.initial_state')
        print(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            print("initial gripper xpos:")
            print(self.initial_gripper_xpos)

            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.current_qpos = self.sim.data.qpos[self.sim.model.jnt_qposadr[6:13]]
        self.initial_state = self.sim.get_state()

        print("done env initialization")

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