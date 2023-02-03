from utils import math_helpers
from policies import ImpalaPolicy, AtariPolicy, DiscretePolicy, MujocoPolicy
from utils import ImpalaEnvWrapper
import gym
import numpy as np


def get_init_data(env_id, random_seed):
    if "procgen" in env_id:
        env = ImpalaEnvWrapper(gym.make(env_id, distribution_mode="easy", num_levels=500))
        policy = ImpalaPolicy(env.observation_space.shape, env.action_space.n, seed=random_seed)
        strategy_distance_fn = math_helpers.categorical_tvd
    elif "NoFrameskip" in env_id:
        from baselines.common import atari_wrappers
        env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari("SpaceInvadersNoFrameskip"), frame_stack=True,
                                                scale=True)
        policy = AtariPolicy(env.observation_space.shape, env.action_space.n, seed=random_seed)
        strategy_distance_fn = math_helpers.categorical_tvd
    else:
        env = gym.make(env_id)
        action_space = env.action_space
        env.seed(random_seed)
        action_space.seed(random_seed)
        n_inputs = np.prod(env.observation_space.shape)
        if type(action_space) == gym.spaces.Discrete:
            policy = DiscretePolicy(n_inputs, action_space.n, seed=random_seed)
            strategy_distance_fn = math_helpers.categorical_tvd
        else:
            policy = MujocoPolicy(n_inputs, np.prod(env.action_space.shape), seed=random_seed)
            strategy_distance_fn = math_helpers.gaussian_wasserstein_dist

    return env, policy, strategy_distance_fn
