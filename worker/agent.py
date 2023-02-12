import numpy as np
from utils.math_helpers import WelfordRunningStat


class Agent(object):
    def __init__(self, policy, env, random_seed,
                 normalize_obs=False,
                 obs_stats_update_chance=0.01,
                 episode_timestep_limit=-1,
                 observation_clip_range=10):

        self.policy = policy
        self.env = env
        self.rng = np.random.RandomState(random_seed)
        self.last_obs = env.reset()
        self.cumulative_timesteps = 0

        if episode_timestep_limit == -1:
            self.ts_limit = np.inf
        else:
            self.ts_limit = episode_timestep_limit

        self.obs_clip = observation_clip_range
        self.obs_stats = WelfordRunningStat(policy.input_shape)
        self.normalize_obs = normalize_obs
        self.obs_stats_update_chance = obs_stats_update_chance

        self.saved_states = []

    def collect_return(self, eval_run=False, save_states=False, mean=1, std=0):
        policy = self.policy
        env = self.env
        obs = self.last_obs
        obs_clip = self.obs_clip

        # Agents produce updates for obs statistics. Learners accumulate these updates in a fixed obs stats object such
        # that every worker connected to a learner should be guaranteed to use the same observation statistics.
        if self.normalize_obs:
            self.obs_stats.reset()

        reward = 0
        steps = 0
        states = []
        policy.reset()
        ts_limit = self.ts_limit
        while steps < ts_limit:
            states.append(obs)
            if self.normalize_obs:
                if self.rng.uniform(0, 1) < self.obs_stats_update_chance:
                    self.obs_stats.increment(obs, 1)
                obs = np.subtract(obs, mean) / std
                obs = np.clip(obs, -obs_clip, obs_clip)

            action = policy.get_action(obs, deterministic=eval_run)
            new_obs, rew, done, _ = env.step(action)

            reward += rew
            steps += 1
            obs = new_obs

            if done:
                obs = env.reset()
                break

        self.last_obs = obs
        self.cumulative_timesteps += steps

        if save_states:
            self.saved_states = states

        states = np.asarray(states)

        if self.normalize_obs:
            states = np.clip((states - mean) / std, -obs_clip, obs_clip)

        entropy = policy.get_entropy(states)
        policy.reset()

        # jiggle the reward a tiny amount just in case all rewards are identical
        reward += self.rng.choice((-1e-12, 1e-12))

        return reward, entropy, steps
