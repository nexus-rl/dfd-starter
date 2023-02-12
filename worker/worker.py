from learner import FDReturn
import torch
import numpy as np
from utils.math_helpers import WelfordRunningStat
from utils import generate_random_id


class Worker(object):
    def __init__(self, policy, agent, noise_source, strategy_handler, sigma=0.02, eval_prob=0.1, random_seed=123, eval_only=False):
        self.policy = policy
        self.agent = agent
        self.noise_source = noise_source
        self.strategy_handler = strategy_handler
        self.sigma = sigma
        self.epoch = -1
        self.rng = np.random.RandomState(random_seed)
        self.eval_prob = eval_prob
        self.fixed_obs_stats = WelfordRunningStat(policy.input_shape)
        self.eval_only = eval_only
        self.worker_id = generate_random_id()

    @torch.no_grad()
    def collect_returns(self, n=1):
        returns = []
        for i in range(n):
            is_eval = self.eval_only or self.rng.uniform(0, 1) < self.eval_prob

            if not is_eval:
                flat = self.policy.get_trainable_flat()
                encoded_perturbation, perturbation = self.noise_source.sample()
                new_flat = flat + self.sigma * perturbation

                self.policy.set_trainable_flat(new_flat)
                ret = self._build_ret(encoded_perturbation, is_eval)
                self.policy.set_trainable_flat(flat)
            else:
                ret = self._build_ret("0", is_eval)
                ret.eval_states = [state for state in self.agent.saved_states]

            returns.append(ret)
        return returns

    def update(self, state):
        self.policy.deserialize(state.policy_params)
        self.epoch = state.epoch
        self.fixed_obs_stats.deserialize(state.obs_stats)

    def _build_ret(self, encoded_perturbation, is_eval):
        ret = FDReturn()
        rew, ent, timesteps = self.agent.collect_return(eval_run=is_eval, save_states=is_eval,
                                                        mean=self.fixed_obs_stats.mean, std=self.fixed_obs_stats.std)
        ret.is_eval = is_eval
        ret.timesteps = timesteps
        ret.encoded_noise = encoded_perturbation
        ret.reward = rew
        ret.novelty = self.strategy_handler.compute_novelty(self.policy)
        ret.entropy = ent
        ret.epoch = self.epoch
        ret.obs_stats_update = self.agent.obs_stats.serialize()
        ret.worker_id = self.worker_id
        return ret
