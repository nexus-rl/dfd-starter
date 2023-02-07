from learner import FiniteDifferences, FDState
from worker import Agent, Worker, GRPCWorker
from utils import SimpleNoiseSource, ImpalaEnvWrapper, AdaptiveOmega, SharedNoiseTable, RNGNoiseSource
from strategy import StrategyHandler
from custom_envs import simple_trap_env
from policies import ImpalaPolicy, DiscretePolicy, AtariPolicy, MujocoPolicy
import gym
import procgen
from utils import math_helpers, init_helper
from dsgd import DSGD
import torch
import numpy as np
import random
import wandb
import time
import uuid


class ServerRunner(object):
    def __init__(self,
                 opt_fn=DSGD,
                 env_id="Walker2d-v2",
                 normalize_obs=True,
                 obs_stats_update_chance=0.01,
                 timestep_limit=50_000_000,
                 learning_rate=0.01,
                 noise_std=0.02,
                 batch_size=40,
                 ent_coef=0.0,
                 random_seed=123,
                 max_delayed_return=100,
                 vbn_buffer_size=0,
                 zeta_size=2,
                 max_strategy_history_size=2,
                 eval_prob=0.05,
                 omega_default_value=1,
                 omega_improvement_threshold=1.035,
                 omega_reward_history_size=20,
                 omega_min_value=0,
                 omega_max_value=1,
                 omega_steps_to_min=25,
                 omega_steps_to_max=75,
                 log_to_wandb=False,
                 existing_wandb_run=None,
                 wandb_project="fd-starter",
                 wandb_group=None,
                 wandb_run_name="dfd_exp_buffer_pert_dist"):

        self.wandb_run = None

        if existing_wandb_run is not None:
            self.wandb_run = existing_wandb_run
        elif log_to_wandb:
            self.wandb_run = wandb.init(project=wandb_project,
                                        group=wandb_group if wandb_group is not None else env_id,
                                        name=wandb_run_name if wandb_run_name is not None else "Seed {}".format(random_seed),
                                        config=None, reinit=True)

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

        self.timestep_limit = timestep_limit
        self.env, self.policy, strategy_distance_fn = init_helper.get_init_data(env_id, random_seed)

        opt = opt_fn(self.policy.parameters(), lr=learning_rate)
        noise_source = RNGNoiseSource(self.policy.num_params, random_seed=random_seed)

        self.strategy_handler = StrategyHandler(self.policy, strategy_distance_fn, max_history_size=max_strategy_history_size)

        self.learner = FiniteDifferences(self.policy, opt, self.omega, noise_source,
                                         noise_std=noise_std,
                                         batch_size=batch_size,
                                         ent_coef=ent_coef,
                                         max_delayed_return=max_delayed_return)

        self.normalize_obs = normalize_obs
        self.policy_reward = 0
        self.policy_entropy = 0
        self.policy_novelty = 0
        self.zeta = []
        self.vbn_buffer = None
        self.global_obs_stats = math_helpers.WelfordRunningStat(self.policy.input_shape)
        self._sample_initial_buffers(vbn_buffer_size)

        self.current_state = FDState()
        self.current_state.experiment_id = uuid.uuid1().hex
        self.current_state.strategy_frames = self.zeta
        self.current_state.strategy_history = self.strategy_handler.strategy_tensor
        self.current_state.policy_params = self.policy.serialize()
        self.current_state.obs_stats = self.global_obs_stats.serialize()
        self.current_state.epoch = self.learner.epoch
        self.current_state.cfg = {"env_id": env_id, "noise_std": noise_std, "normalize_obs": self.normalize_obs,
                                  "obs_stats_update_chance": obs_stats_update_chance, "random_seed": random_seed,
                                  "eval_prob": eval_prob, "max_strategy_history_size": max_strategy_history_size}

        self.worker = GRPCWorker(self.current_state)

    @torch.no_grad()
    def train(self):
        cumulative_timesteps = 0
        current_state = self.current_state
        policy = self.policy
        learner = self.learner
        worker = self.worker
        batch_size = self.batch_size
        strategy_handler = self.strategy_handler
        global_obs_stats = self.global_obs_stats
        zeta = self.zeta
        ts_limit = self.timestep_limit
        idxs = [i for i in range(len(zeta))]
        max_delayed_return = self.learner.max_delayed_return
        strategy_handler.add_policy(policy)
        worker.update(current_state)

        worker.start(address="localhost", port=1025)

        while cumulative_timesteps < ts_limit:
            t1 = time.perf_counter()
            ret_rewards = []
            ret_novelties = []
            non_eval_returns = []
            any_eval = False

            returns, timesteps, n_delayed, n_discarded = worker.collect_returns(batch_size=batch_size,
                                                                                current_epoch=learner.epoch,
                                                                                max_delayed_return=max_delayed_return)
            print("received",len(returns))
            self.learner.discarded_returns += n_discarded
            cumulative_timesteps += timesteps

            for ret in returns:
                global_obs_stats.increment_from_obs_stats_update(ret.obs_stats_update)
                if ret.is_eval:
                    any_eval = True
                    self.policy_reward = self.policy_reward * 0.9 + ret.reward * 0.1
                    self.policy_entropy = self.policy_entropy * 0.9 + ret.entropy * 0.1
                    self.policy_novelty = self.policy_novelty * 0.9 + ret.novelty * 0.1
                    self.rng.shuffle(idxs)
                    zeta[idxs[:len(ret.eval_states)]] = ret.eval_states[:self.zeta_size]
                else:
                    non_eval_returns.append(ret)
                    ret_rewards.append(ret.reward)
                    ret_novelties.append(ret.novelty)

            # print("collected {} returns of which {} are delayed.".format(len(returns), n_delayed))
            if any_eval:
                strategy_handler.set_zeta(zeta)
                if len(ret_rewards) != 0:
                    self.omega.step(np.mean(ret_rewards))

            update_magnitude = learner.step(non_eval_returns, self.policy_reward, self.policy_novelty, self.policy_entropy)

            if self.vbn_buffer is not None:
                self.policy.compute_vbn(self.vbn_buffer)

            if update_magnitude > 0 and len(ret_rewards) != 0:
                strategy_handler.add_policy(policy)
                delayed_ratio = n_delayed / len(non_eval_returns)

                epoch_time = time.perf_counter() - t1
                epoch_report = {"Epoch":                learner.epoch,
                                "Epoch Time":           epoch_time,
                                "Cumulative Timesteps": cumulative_timesteps,
                                "\nPolicy Reward":      self.policy_reward,
                                "Policy Entropy":       self.policy_entropy,
                                "Policy Novelty":       self.policy_novelty,
                                "\nNoisy Reward":       np.mean(ret_rewards),
                                "Noisy Novelty":        np.mean(ret_novelties),
                                "\nDelayed Ratio":      delayed_ratio,
                                "Update Magnitude":     update_magnitude,
                                "Omega":                self.omega.omega,
                                "Discarded Returns":    learner.discarded_returns}
                self._report_epoch(epoch_report)

            current_state.strategy_frames = zeta
            current_state.strategy_history = strategy_handler.strategy_tensor
            current_state.policy_params = policy.serialize()
            current_state.epoch = learner.epoch
            current_state.obs_stats = global_obs_stats.serialize()
            worker.update(current_state)

        worker.stop()
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
    def _sample_initial_buffers(self, vbn_buffer_size):
        self.vbn_buffer = []
        self.zeta = []
        obs = self.env.reset()
        for i in range(max(vbn_buffer_size, self.zeta_size)):
            if self.normalize_obs:
                self.global_obs_stats.increment(obs, 1)

            if i < self.zeta_size:
                self.zeta.append(obs)

            if vbn_buffer_size > 0 and i < vbn_buffer_size:
                self.vbn_buffer.append(obs)

            obs, rew, done, _ = self.env.step(self.env.action_space.sample())
            if done:
                obs = self.env.reset()

        self.vbn_buffer = np.asarray(self.vbn_buffer)
        self.zeta = np.asarray(self.zeta)


def main():
    runner = ServerRunner()
    runner.train()


def sweep():
    def sweep_fn():
        run = wandb.init(project="unnamed-sweep")
        runner = ServerRunner(log_to_wandb=True,
                              existing_wandb_run=run,
                              learning_rate=run.config.learning_rate,
                              noise_std=run.config.noise_std,
                              batch_size=run.config.batch_size
                              )
        runner.train()

    sweep_config = \
        {
            "method": "random",
            'metric': {
                'goal': 'maximize',
                'name': 'Policy Reward'
            },
            "parameters":
                {
                    "learning_rate":
                        {
                            "values": [0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
                        },
                    "noise_std":
                        {
                            "values": [0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
                        },
                    "batch_size":
                        {
                            "values": [10, 25, 50, 75, 100]
                        }
                }
        }

    sweep_id = "053meetx"  # wandb.sweep(sweep=sweep_config, project='unnamed-sweep')
    wandb.agent(sweep_id, function=sweep_fn, count=180, project="mujoco-sweep-longer")


if __name__ == "__main__":
    main()
    # sweep()
