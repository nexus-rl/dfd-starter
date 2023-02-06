import time

from learner import FiniteDifferences, FDState
from worker import Agent, Worker
from utils import SimpleNoiseSource, ImpalaEnvWrapper, AdaptiveOmega, SharedNoiseTable, RNGNoiseSource
from strategy import StrategyHandler
from policies import ImpalaPolicy, DiscretePolicy, AtariPolicy, MujocoPolicy
import gym
from utils import math_helpers
from dsgd import DSGD
import torch
import numpy as np
import random
import wandb


class SequentialRunner(object):
    def __init__(self,
                 opt_fn=DSGD,
                 env_id="Walker2d-v4",
                 normalize_obs=False,
                 learning_rate=0.01,
                 noise_std=0.02,
                 batch_size=40,
                 ent_coef=0.0,
                 random_seed=123,
                 max_delayed_return=10,
                 vbn_buffer_size=0,
                 zeta_size=200,
                 max_strategy_history_size=200,
                 eval_prob=0.05,
                 omega_default_value=0,
                 omega_improvement_threshold=1.035,
                 omega_reward_history_size=20,
                 omega_min_value=0,
                 omega_max_value=1,
                 omega_steps_to_min=25,
                 omega_steps_to_max=75,
                 log_to_wandb=False,
                 wandb_project="fd-starter",
                 wandb_group=None,
                 wandb_run_name=None):

        if log_to_wandb:
            self.wandb_run = wandb.init(project=wandb_project,
                                        group=wandb_group if wandb_group is not None else env_id,
                                        name=wandb_run_name if wandb_run_name is not None else "Seed {}".format(random_seed),
                                        config=None,
                                        reinit=True)
        else:
            self.wandb_run=None

        self.rng = np.random.RandomState(random_seed)
        self.omega = AdaptiveOmega(default_value=omega_default_value,
                                   improvement_threshold=omega_improvement_threshold,
                                   reward_history_size=omega_reward_history_size,
                                   min_value=omega_min_value,
                                   max_value=omega_max_value,
                                   steps_to_min=omega_steps_to_min,
                                   steps_to_max=omega_steps_to_max)

        self.batch_size = batch_size
        self.zeta_size = zeta_size
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        if "procgen" in env_id:
            self.env = ImpalaEnvWrapper(gym.make(env_id, distribution_mode='easy'))
            self.policy = ImpalaPolicy(self.env.observation_space.shape, self.env.action_space.n, seed=random_seed)
            strategy_distance_fn = math_helpers.categorical_tvd
        else:
            self.env = gym.make(env_id)
            action_space = self.env.action_space
            self.env.reset(seed=random_seed)
            action_space.seed(random_seed)
            n_inputs = np.prod(self.env.observation_space.shape)

            if type(action_space) == gym.spaces.Discrete:
                self.policy = DiscretePolicy(n_inputs, action_space.n, seed=random_seed)
                strategy_distance_fn = math_helpers.categorical_tvd
            else:
                self.policy = MujocoPolicy(n_inputs, np.prod(self.env.action_space.shape), seed=random_seed)
                strategy_distance_fn = math_helpers.gaussian_wasserstein_dist

        opt = opt_fn(self.policy.parameters(), lr=learning_rate)
        # noise_source = SimpleNoiseSource(self.policy.num_params, random_seed=random_seed)
        # noise_source = SharedNoiseTable(25000000, self.policy.num_params, random_seed=random_seed)
        noise_source = RNGNoiseSource(self.policy.num_params, random_seed=random_seed)

        self.strategy_handler = StrategyHandler(self.policy, strategy_distance_fn, max_history_size=max_strategy_history_size)
        self.agent = Agent(self.policy, self.env, random_seed, normalize_obs=normalize_obs)

        self.worker = Worker(self.policy, self.agent, noise_source, self.strategy_handler, sigma=noise_std,
                             random_seed=random_seed, eval_prob=eval_prob)

        self.learner = FiniteDifferences(self.policy, opt, self.omega, noise_source, ent_coef, max_delayed_return)
        self.policy_reward = 0
        self.policy_entropy = 0
        self.policy_novelty = 0
        self.zeta = []
        self.vbn_buffer = None
        self._sample_initial_buffers(vbn_buffer_size)

        self.current_state = FDState()
        self.current_state.strategy_frames = self.zeta
        self.current_state.strategy_history = self.strategy_handler.strategy_tensor
        self.current_state.policy_params = self.policy.get_trainable_flat()
        self.current_state.epoch = 0
        self.current_state.experiment_id = 1234
        self.current_state.cfg = {"key1": 1, "key2": 2, "key3": 3, "key4": {"inner_key1": 41}}

    @torch.no_grad()
    def train(self, n_epochs):
        policy = self.policy
        learner = self.learner
        worker = self.worker
        agent = self.agent
        batch_size = self.batch_size
        strategy_handler = self.strategy_handler
        current_state = self.current_state
        zeta = self.zeta
        idxs = [i for i in range(len(zeta))]

        strategy_handler.add_policy(policy)
        worker.update(current_state)

        for epoch in range(n_epochs):
            t1 = time.perf_counter()
            ret_rewards = []
            ret_novelties = []
            rets = []
            any_eval = False
            while len(rets) < batch_size:
                returns = worker.collect_returns()
                for ret in returns:
                    if ret.is_eval:
                        any_eval = True
                        self.policy_reward = self.policy_reward * 0.9 + ret.reward * 0.1
                        self.policy_entropy = self.policy_entropy * 0.9 + ret.entropy * 0.1
                        self.policy_novelty = self.policy_novelty * 0.9 + ret.novelty * 0.1
                        self.rng.shuffle(idxs)
                        zeta[idxs[:len(ret.eval_states)]] = ret.eval_states[:self.zeta_size]
                    else:
                        rets.append(ret)
                        ret_rewards.append(ret.reward)
                        ret_novelties.append(ret.novelty)

            if any_eval:
                strategy_handler.set_zeta(zeta)
                self.omega.step(np.mean(ret_rewards))
                # self.omega.step(np.mean(ret_rewards))

            update_magnitude = learner.step(rets, self.policy_reward, self.policy_novelty, self.policy_entropy)

            if self.vbn_buffer is not None:
                self.policy.compute_vbn(self.vbn_buffer)

            if update_magnitude > 0:
                strategy_handler.add_policy(policy)
                current_state.strategy_frames = zeta
                current_state.strategy_history = strategy_handler.strategy_tensor
                current_state.policy_params = policy.get_trainable_flat()
                current_state.epoch = learner.epoch

                worker.update(current_state)
                epoch_time = time.perf_counter() - t1
                epoch_report = {"Epoch": learner.epoch,
                                "Epoch Time": epoch_time,
                                "Cumulative Timesteps": agent.cumulative_timesteps,
                                "Policy Reward":        self.policy_reward,
                                "Policy Entropy":       self.policy_entropy,
                                "Policy Novelty":       self.policy_novelty,
                                "Noisy Reward":         np.mean(ret_rewards),
                                "Noisy Novelty":        np.mean(ret_novelties),
                                "Update Magnitude":     update_magnitude,
                                "Omega":                self.omega.omega}

                self._report_epoch(epoch_report)

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _report_epoch(self, epoch_report):
        if self.wandb_run is not None:
            self.wandb_run.log(epoch_report)

        print("\n***********Begin Epoch Report***********")
        for key, val in epoch_report.items():
            if key[0] == "_":
                continue
            if type(val) in (float, np.float32, np.float64):
                print("{} {:7.4f}".format(key, val))
            else:
                print(key, val)
        print("***********End Epoch Report***********")

    @torch.no_grad()
    def _sample_initial_buffers(self, buffer_size):
        self.vbn_buffer = []
        self.zeta = []
        obs, _ = self.env.reset()
        for i in range(max(buffer_size, self.zeta_size)):
            if i < self.zeta_size:
                self.zeta.append(obs)
            if buffer_size > 0 and i < buffer_size:
                self.vbn_buffer.append(obs)
            obs, rew, terminated, truncated, _ = self.env.step(self.env.action_space.sample())
            if terminated or truncated:
                obs, _ = self.env.reset()

        self.vbn_buffer = np.asarray(self.vbn_buffer)
        self.zeta = np.asarray(self.zeta)
