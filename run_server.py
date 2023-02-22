from learner import FiniteDifferences, FDState
from worker import Agent, Worker, GRPCWorker
from utils import SimpleNoiseSource, ImpalaEnvWrapper, AdaptiveOmega, SharedNoiseTable, RNGNoiseSource
from strategy import StrategyHandler
from custom_envs import simple_trap_env, rocketsim
from policies import ImpalaPolicy, DiscretePolicy, AtariPolicy, MujocoPolicy
import gym
from utils import math_helpers, init_helper
from dsgd import DSGD
import torch
import numpy as np
import random
import wandb
import time
import uuid
import datetime


class ServerRunner(object):
    def __init__(self,
                 env,
                 env_id,
                 policy,
                 strategy_distance_fn,
                 opt_fn=DSGD,
                 normalize_obs=True,
                 obs_stats_update_chance=0.01,
                 timestep_limit=None,
                 learning_rate=0.01,
                 noise_std=0.02,
                 batch_size=40,
                 ent_coef=0.0,
                 random_seed=123,
                 max_delayed_return=10,
                 vbn_buffer_size=0,
                 collect_zeta=True,
                 zeta_size=100,
                 max_strategy_history_size=0,
                 eval_prob=0.05,
                 episode_timestep_limit=-1,
                 observation_clip_range=10,
                 omega_default_value=0,
                 omega_improvement_threshold=1.035,
                 omega_reward_history_size=20,
                 omega_min_value=0,
                 omega_max_value=1,
                 omega_steps_to_min=25,
                 omega_steps_to_max=75,
                 wandb_run=None,
                 bind_port=None,
                 bind_address="localhost"):
        
        self.env = env
        self.policy = policy
        self.strategy_distance_fn = strategy_distance_fn

        self.wandb_run = wandb_run

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

        opt = opt_fn(self.policy.parameters(), lr=learning_rate)
        noise_source = RNGNoiseSource(self.policy.num_params, random_seed=random_seed)

        self.strategy_handler = StrategyHandler(self.policy, strategy_distance_fn, max_history_size=max_strategy_history_size)

        self.learner = FiniteDifferences(self.policy, opt, self.omega, noise_source,
                                         noise_std=noise_std,
                                         batch_size=batch_size,
                                         ent_coef=ent_coef,
                                         max_delayed_return=max_delayed_return)

        self.policy_reward = None
        self.policy_entropy = None
        self.policy_novelty = None
        self.zeta = []
        self.vbn_buffer = None
        self.normalize_obs = normalize_obs
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
                                  "eval_prob": eval_prob, "max_strategy_history_size": max_strategy_history_size,
                                  "observation_clip_range":observation_clip_range,
                                  "episode_timestep_limit": episode_timestep_limit,
                                  "collect_zeta": collect_zeta and self.zeta_size > 0}

        self.worker = GRPCWorker(self.current_state)

        self.bind_address = bind_address
        self.bind_port = bind_port

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

        worker.start(address=self.bind_address, port=self.bind_port)
        t1 = time.perf_counter()

        n_waiting_last = 0

        while ts_limit is None or cumulative_timesteps < ts_limit:
            ret_rewards = []
            ret_novelties = []
            non_eval_returns = []
            any_eval = False

            consumed_steps_per_sec = 0
            produced_steps_per_sec = 0

            returns, timesteps, n_delayed, n_discarded, n_waiting = worker.collect_returns(batch_size=batch_size,
                                                                                current_epoch=learner.epoch,
                                                                                max_delayed_return=max_delayed_return)
            # print("received",len(returns))
            consumed_timesteps = timesteps

            # I'm ignoring delayed here because those are included in
            # n_timesteps. We also subtract the number of timesteps that were
            # hanging out in the buffer at the start of the last epoch
            produced_timesteps = timesteps + n_discarded + n_waiting - n_waiting_last

            # keep track of how many are waiting now so we can subtract them
            # from the next epoch's count
            n_waiting_last = n_waiting

            worker_count = len(set(r.worker_id for r in returns))

            self.learner.discarded_returns += n_discarded
            cumulative_timesteps += timesteps

            for ret in returns:
                if self.normalize_obs:
                    global_obs_stats.increment_from_obs_stats_update(ret.obs_stats_update)

                if ret.is_eval:
                    any_eval = True
                    if self.policy_reward is None:
                        self.policy_reward = ret.reward
                        self.policy_entropy = ret.entropy
                        self.policy_novelty = ret.novelty
                    else:
                        self.policy_reward = self.policy_reward * 0.9 + ret.reward * 0.1
                        self.policy_entropy = self.policy_entropy * 0.9 + ret.entropy * 0.1
                        self.policy_novelty = self.policy_novelty * 0.9 + ret.novelty * 0.1
                    self.rng.shuffle(idxs)
                    if self.zeta_size > 0 and len(ret.eval_states) > 0:
                        zeta[idxs[:len(ret.eval_states)]] = ret.eval_states[:self.zeta_size]
                else:
                    non_eval_returns.append(ret)
                    ret_rewards.append(ret.reward)
                    ret_novelties.append(ret.novelty)

            if any_eval:
                strategy_handler.set_zeta(zeta)
                self.omega.step(self.policy_reward)
                # if len(ret_rewards) != 0:
                #     self.omega.step(np.mean(ret_rewards))

            update_magnitude = learner.step(non_eval_returns, self.policy_reward, self.policy_novelty, self.policy_entropy)

            if self.vbn_buffer is not None:
                self.policy.compute_vbn(self.vbn_buffer)

            if update_magnitude > 0 and len(ret_rewards) != 0:
                strategy_handler.add_policy(policy)
                delayed_ratio = n_delayed / len(non_eval_returns)

                epoch_time = time.perf_counter() - t1
                t1 = time.perf_counter()

                consumed_steps_per_sec = consumed_timesteps / epoch_time
                produced_steps_per_sec = produced_timesteps / epoch_time

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
                                "Discarded Returns":    learner.discarded_returns,
                                "\nConsumed Steps": consumed_timesteps,
                                "Produced Steps":   produced_timesteps,
                                "Consumed Steps per Second": consumed_steps_per_sec,
                                "Produced Steps per Second":   produced_steps_per_sec,
                                "\nWorker Count":       worker_count,
                               }
                self._report_epoch(epoch_report)

            current_state.strategy_frames = zeta
            current_state.strategy_history = strategy_handler.strategy_tensor
            current_state.policy_params = policy.serialize()
            current_state.epoch = learner.epoch
            if self.normalize_obs:
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
        obs, _ = self.env.reset()
        for i in range(max(vbn_buffer_size, self.zeta_size)):
            if self.normalize_obs:
                self.global_obs_stats.increment(obs, 1)

            if i < self.zeta_size:
                self.zeta.append(obs)

            if vbn_buffer_size > 0 and i < vbn_buffer_size:
                self.vbn_buffer.append(obs)
            obs, rew, terminated, truncated, _ = self.env.step(self.env.action_space.sample())
            if terminated or truncated:
                obs, _ = self.env.reset()

        self.vbn_buffer = np.asarray(self.vbn_buffer)
        self.zeta = np.asarray(self.zeta)

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


def train(args):

    KNOWN_OPTIMIZERS = {
        "DSGD": lambda: DSGD,
        "Adam": lambda: torch.optim.Adam,
        "SGD": lambda: torch.optim.SGD,
    }

    if args.optimizer not in KNOWN_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {args.optimizer}, must be one of {KNOWN_OPTIMIZERS.keys()}")
    opt_fn = KNOWN_OPTIMIZERS[args.optimizer]()

    if args.env_id not in gym.envs.registry:
        raise ValueError("Unknown env: {}, make sure it is registered to Gym.".format(args.env_id))

    if args.log_to_wandb and args.existing_wandb_run is None:
        # Expect a project name to be passed in
        if args.wandb_project is None:
            raise ValueError("Must pass in a wandb project name (--wandb_project) if not resuming a run")
        # Defaults for group and run name if not passed in
        if args.wandb_group is None:
            args.wandb_group = args.env_id
        if args.wandb_run_name is None:
            start_timestamp = datetime.datetime.utcnow().isoformat(timespec="seconds")
            args.wandb_run_name = f"seed-{args.random_seed}-start-{start_timestamp}"

    wandb_run = None

    if args.log_to_wandb:
        wandb_run = wandb.init(project=args.wandb_project if args.existing_wandb_run is None else None,
                               group=args.wandb_group if args.existing_wandb_run is None else None,
                               name=args.wandb_run_name if args.existing_wandb_run is None else None,
                               id=args.existing_wandb_run,
                               config=vars(args),
                               reinit=args.existing_wandb_run is not None)

    if not args.log_to_wandb:
        print("INFO: Not logging to wandb.")

    env, policy, strategy_distance_fn, action_space = init_helper.get_init_data(args.env_id, args.random_seed)

    if type(action_space) == gym.spaces.Discrete:
        args.normalize_obs = False
    else:
        args.vbn_buffer_size = 0


    runner = ServerRunner(env=env,
                          env_id=args.env_id,
                          policy=policy,
                          strategy_distance_fn=strategy_distance_fn,
                          opt_fn=opt_fn,
                          normalize_obs=args.normalize_obs,
                          obs_stats_update_chance=args.obs_stats_update_chance,
                          learning_rate=args.learning_rate,
                          noise_std=args.noise_std,
                          batch_size=args.batch_size,
                          ent_coef=args.ent_coef,
                          random_seed=args.random_seed,
                          max_delayed_return=args.max_delayed_return,
                          vbn_buffer_size=args.vbn_buffer_size,
                          collect_zeta=args.collect_zeta,
                          zeta_size=args.zeta_size,
                          max_strategy_history_size=args.max_strategy_history_size,
                          eval_prob=args.eval_prob,
                          omega_default_value=args.omega_default_value,
                          omega_improvement_threshold=args.omega_improvement_threshold,
                          omega_reward_history_size=args.omega_reward_history_size,
                          omega_min_value=args.omega_min_value,
                          omega_max_value=args.omega_max_value,
                          omega_steps_to_min=args.omega_steps_to_min,
                          omega_steps_to_max=args.omega_steps_to_max,
                          wandb_run=wandb_run,
                          bind_port=args.bind_port,
                          bind_address=args.bind_address,
                          timestep_limit=args.timestep_limit,
                          episode_timestep_limit=args.episode_timestep_limit if args.episode_timestep_limit is not None else -1,
                          )
    runner.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    command_group = parser.add_argument_group("Command")
    command_group.add_argument("operation", type=str, choices=["train", "sweep"], help="Operation to perform")

    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--env", type=str, dest="env_id", help="The name of the Gym environment on which to train", required=True)
    env_group.add_argument("--episode_timestep_limit", type=int, default=None, help="If set, enforces a limit on the number of timesteps involved in each episode. (default: %(default)s)")

    exp_group = parser.add_argument_group("Exploration")
    exp_group.add_argument("--explore", action=argparse.BooleanOptionalAction, default=False, help="Enable advanced exploration (default: %(default)s)")
    exp_group.add_argument("--collect_zeta",  action="store_true", default=False, help="Collect zeta data for the advanced exploration algorithm (default: %(default)s)")
    exp_group.add_argument("--zeta_size", type=int, default=100, help="Size of the zeta buffer (default: %(default)s)")
    exp_group.add_argument("--max_strategy_history_size", type=int, default=0, help="Maximum size of the strategy history (default: %(default)s)")

    exp_group.add_argument("--omega_default_value", type=float, default=0)
    exp_group.add_argument("--omega_improvement_threshold", type=float, default=1.035)
    exp_group.add_argument("--omega_reward_history_size", type=int, default=20)
    exp_group.add_argument("--omega_min_value", type=float, default=0)
    exp_group.add_argument("--omega_max_value", type=float, default=1)
    exp_group.add_argument("--omega_steps_to_min", type=int, default=25)
    exp_group.add_argument("--omega_steps_to_max", type=int, default=75)

    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--timestep_limit", type=int, default=None, help="Train until this many timesteps (default: %(default)s)")
    training_group.add_argument("--seed", type=int, default=124, dest="random_seed", help="Random seed (default: %(default)s)")
    training_group.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (default: %(default)s)")
    training_group.add_argument("--noise_std", type=float, default=0.02, help="Noise standard deviation (default: %(default)s)")
    training_group.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    training_group.add_argument("--optimizer", type=str, default="DSGD", choices=["DSGD", "Adam", "SGD"], help="Optimizer (default: %(default)s)")
    training_group.add_argument("--max_delayed_return", type=int, default=10, help="Maximum number of steps to delay the return (default: %(default)s)")
    training_group.add_argument("--ent_coef", type=float, default=0, help="Entropy coefficient (default: %(default)s)")
    training_group.add_argument("--vbn_buffer_size", type=int, default=1000, help="Size of the VBN buffer. Ignored when using an environment with a continuous action space. (default: %(default)s)")

    training_group.add_argument('--normalize_obs', action=argparse.BooleanOptionalAction, default=True, help="Normalize observations. Ignored when using an environment with a discrete action space. (default: %(default)s)")
    training_group.add_argument("--obs_stats_update_chance", type=float, default=0.01, help="Probability of updating the observation statistics for normalization. Ignored when using an environment with a discrete action space. (default: %(default)s)")

    training_group.add_argument("--eval_prob", type=float, default=0.05, help="Probability of evaluating the policy (default: %(default)s)")

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument("--log_to_wandb", action="store_true", help="Log to wandb (default: %(default)s)")
    logging_group.add_argument("--wandb_project", type=str, default=None, help="Name of the wandb project (default: %(default)s)")
    logging_group.add_argument("--wandb_group", type=str, default=None, help="Name of the wandb group (default: %(default)s)")
    logging_group.add_argument("--wandb_run_name", type=str, default=None, help="Name of the wandb run (default: %(default)s)")
    logging_group.add_argument("--existing_wandb_run", type=str, default=None, help="The ID of an existing wandb run to log to (default: %(default)s)")

    net_group = parser.add_argument_group("Network")
    net_group.add_argument("--bind_port", type=int, default=1025, help="Port to bind to (default: %(default)s)")
    net_group.add_argument("--bind_address", type=str, default="localhost", help="Address to bind to (default: %(default)s)")

    args = parser.parse_args()
    print('Args', args)

    if not args.explore:
        args.collect_zeta = False
        args.zeta_size = 0
        if args.optimizer != "DSGD":
            args.omega_reward_history_size = 0


    if args.operation == "train":
        train(args)
    elif args.operation == "sweep":
        raise NotImplementedError("Sweep not implemented yet, someone should do that.")
