from baselines.common.vec_env import VecEnv
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np

class VecNormalize(VecEnv):
    """
    Vectorized environment base class
    """
    def __init__(self, venv,
                 ob_mean=None, ob_var=None, ob_count=None,
                 ret_mean=None, ret_var=None, ret_count=None,
                 ob=True, ret=True,
                 clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        self.venv = venv
        self._observation_space = self.venv.observation_space
        self._action_space = venv.action_space
        self.ob_rms = RunningMeanStd(shape=self._observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        if ob and not(ob_mean is None):
            self.ob_rms.mean = ob_mean
            self.ob_rms.var = ob_var
            self.ob_rms.count = ob_count
        
        if ret and not(ret_mean is None):
            self.ret_rms.mean = ret_mean
            self.ret_rms.var = ret_var
            self.ret_rms.count = ret_count

        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step(vac)
        obs = self._obfilt(obs)
        return obs, rews, news, infos

    def normalize_rew(self, rews):
        self.ret = self.ret * self.gamma + rews
        if self.ret_rms: 
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return rews        

    def _obfilt(self, obs):
        if self.ob_rms: 
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs
    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)
    @property
    def action_space(self):
        return self._action_space
    @property
    def observation_space(self):
        return self._observation_space
    def close(self):
        self.venv.close()
    @property
    def num_envs(self):
        return self.venv.num_envs


class VecNormalizeTest(VecEnv):
    def __init__(self, venv, mean, var, clipob=10., epsilon=1e-8):
        self.venv = venv
        self._observation_space = self.venv.observation_space
        self._action_space = venv.action_space

        self.mean = mean
        self.var = var
        self.clipob = clipob
        self.epsilon = epsilon

    def render(self):
        return self.venv.render()

    def step(self, vac):
        obs, rews, dones, infos = self.venv.step(vac)
        self.origin_obs = obs
        obs = self._obfilt(obs)
        return obs, rews, dones, infos

    def normalize_rew(self, rews):
        return rews     

    def _obfilt(self, obs):
        obs = np.clip((obs - self.mean) / np.sqrt(self.var + self.epsilon), -self.clipob, self.clipob)
        return obs

    def reset(self):
        obs = self.venv.reset()
        self.origin_obs = obs
        return self._obfilt(obs)

    @property
    def action_space(self):
        return self._action_space
    @property
    def observation_space(self):
        return self._observation_space
    def close(self):
        self.venv.close()
    @property
    def num_envs(self):
        return self.venv.num_envs
    
