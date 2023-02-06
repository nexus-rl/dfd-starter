from strategy import StrategyHandler
from worker import Agent
from policies import MujocoPolicy, DiscretePolicy
import gym
import torch
import random
import numpy as np


def run_test():
    env_id = "LunarLander-v2"
    random_seed = 123

    env = gym.make(env_id)

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    env.seed(random_seed)
    env.action_space.seed(random_seed)
    n_inputs = np.prod(env.observation_space.shape)
    # policy = MujocoPolicy(n_inputs, np.prod(env.action_space.shape))
    policy = DiscretePolicy(n_inputs, env.action_space.n)

    handler = StrategyHandler(policy)

    obs, _ = env.reset()
    zeta = [obs]
    done = False
    for i in range(200-1):
        action = policy.get_action(zeta[-1])
        obs, rew, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        zeta.append(obs)
    zeta = np.asarray(zeta)

    handler.set_zeta(zeta)
    # for i in range(25):
    #     handler.add_policy(MujocoPolicy(n_inputs, np.prod(env.action_space.shape)).get_trainable_flat())
    # other = MujocoPolicy(n_inputs, np.prod(env.action_space.shape))
    other = DiscretePolicy(n_inputs, env.action_space.n)

    zeros = np.zeros(other.num_params)
    other.set_trainable_flat(zeros)
    handler.add_policy(other.get_trainable_flat())

    for i in range(100):
        flat = other.get_trainable_flat()
        flat = flat + np.random.randn(other.num_params)*0.1
        other.set_trainable_flat(flat)
        handler.add_policy(flat)
        handler.evaluate_strategies()
        handler.compute_novelty(policy.get_trainable_flat())

if __name__ == "__main__":
    run_test()
