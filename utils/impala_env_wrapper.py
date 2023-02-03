import gym
import torch


class ImpalaEnvWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Input shape to impala is (batch, time, channels, width, height)
        self.obs_shape = [1, 1, self._env.observation_space.shape[-1]] + [arg for arg in self._env.observation_space.shape[:-1]]

    def step(self, action):
        obs, rew, done, etc = self._env.step(action)
        return self._format_obs(obs, rew, done), rew, done, etc

    def reset(self, *, seed=None, return_info=False, options=None):
        return self._format_obs(self._env.reset(), 0, False)

    def render(self, mode="human"):
        return self._env.render(mode)

    def _format_obs(self, obs, rew, done):
        return {"frame": torch.as_tensor(obs, dtype=torch.float32).view(self.obs_shape),
                "reward": torch.as_tensor(rew, dtype=torch.float32).view(1, 1),
                "done": torch.as_tensor(1 if done else 0, dtype=torch.bool).view(1, 1)}