import time
from custom_envs import simple_trap_env, rocketsim
from networking.client import RPCClient
from learner import FDState
from worker import Agent, Worker
from utils import SimpleNoiseSource, ImpalaEnvWrapper, AdaptiveOmega, SharedNoiseTable, RNGNoiseSource
from strategy import StrategyHandler
from policies import ImpalaPolicy, DiscretePolicy, AtariPolicy, MujocoPolicy
import gym
from utils import math_helpers, init_helper
import torch
import numpy as np
import random
import os
from multiprocessing import Process
from functools import partial

torch.set_num_threads(1)


class ClientRunner(object):
    def __init__(self, eval_only=False, render=False, server_port=None, server_address="localhost"):
        self.policy_reward = 0
        self.policy_entropy = 0
        self.policy_novelty = 0
        self.worker = None
        self.strategy_handler = None
        self.policy = None
        self.env = None
        self.rng = None
        self.worker_id = ""
        self.eval_only = eval_only
        self.render = render
        self.server_port = server_port
        self.server_address = server_address

        self.client = RPCClient()

    @torch.no_grad()
    def run(self):
        client = self.client
        client.connect(address=self.server_address, port=self.server_port)
        self.receive_config()

        policy = self.policy
        worker = self.worker

        policy.deserialize(client.current_state.policy_params)
        strategy_handler = self.strategy_handler
        worker.update(client.current_state)
        strategy_handler.add_policy(policy, obs_stats=(worker.fixed_obs_stats.mean, worker.fixed_obs_stats.std))

        running = True

        while running:
            returns = []
            t1 = time.time()
            while time.time() - t1 < 0.01:
                returns += worker.collect_returns()

            # print("Submitted {} returns.".format(len(returns)))
            client.submit_returns(returns)

            status = client.get_server_state()
            if status == RPCClient.NEW_STATE_FLAG:
                state = client.current_state
                worker.update(state)

            elif status == RPCClient.RPC_FAILED_FLAG:
                print("FAILED TO GET UPDATE FROM SERVER")
                n_attempts = 60
                running = False

                for i in range(n_attempts):
                    print("Retrying server... {}/{}".format(i+1, n_attempts))
                    time.sleep(1)
                    status = client.get_server_state()
                    print(status)
                    if status != RPCClient.RPC_FAILED_FLAG:
                        print("Connection re-established!")
                        running = True
                        break

                if not running:
                    import sys
                    print("FAILED TO RE-ESTABLISH CONNECTION WITH SERVER --- PROGRAM TERMINATING")
                    sys.exit(-1)

                worker.update(client.current_state)
            # Don't use elif here. If the connection fails and is then re-established, new_experiment_flag may be set.
            if status == RPCClient.NEW_EXPERIMENT_FLAG:
                print("CLIENT GOT NEW EXPERIMENT")
                cfg = client.current_state.cfg
                self.configure(cfg["env_id"], cfg["normalize_obs"], cfg["obs_stats_update_chance"], cfg["noise_std"],
                               cfg["random_seed"], cfg["eval_prob"], cfg["max_strategy_history_size"],
                               cfg["observation_clip_range"], cfg["episode_timestep_limit"], cfg["collect_zeta"])

                strategy_handler = self.strategy_handler
                policy = self.policy
                worker = self.worker

                policy.deserialize(client.current_state.policy_params)
                strategy_handler.add_policy(policy, obs_stats=(worker.fixed_obs_stats.mean, worker.fixed_obs_stats.std))
                worker.update(client.current_state)

        client.disconnect()

    def receive_config(self):
        status = self.client.get_server_state()
        while status != RPCClient.NEW_EXPERIMENT_FLAG:
            print("Receiving config...")
            time.sleep(1)
            status = self.client.get_server_state()

        state = self.client.current_state
        cfg = state.cfg
        self.configure(cfg["env_id"], cfg["normalize_obs"], cfg["obs_stats_update_chance"], cfg["noise_std"],
                               cfg["random_seed"], cfg["eval_prob"], cfg["max_strategy_history_size"],
                               cfg["observation_clip_range"], cfg["episode_timestep_limit"], cfg["collect_zeta"])

    def configure(self, env_id="Walker2d-v2", normalize_obs=True, obs_stats_update_chance=0.01,
                  noise_std=0.02, random_seed=124, eval_prob=0.05, max_strategy_history_size=200,
                  observation_clip_range=10, episode_timestep_limit=-1, collect_zeta=False):

        print("Env: {}\nSeed: {}\nNoise std: {}\nNormalize obs: {}".format(env_id, random_seed, noise_std, normalize_obs))
        self.worker_id = "{}".format(random_seed)
        random_seed = int(random_seed)
        self.rng = np.random.RandomState(random_seed)

        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.env, self.policy, strategy_distance_fn, action_space = init_helper.get_init_data(env_id, random_seed, render_mode="human" if self.render else None)

        noise_source = RNGNoiseSource(self.policy.num_params, random_seed=random_seed)

        self.strategy_handler = StrategyHandler(self.policy, strategy_distance_fn,
                                                max_history_size=max_strategy_history_size)

        agent = Agent(self.policy, self.env, random_seed,
                      normalize_obs=normalize_obs,
                      obs_stats_update_chance=obs_stats_update_chance,
                      observation_clip_range=observation_clip_range,
                      episode_timestep_limit=episode_timestep_limit)

        self.worker = Worker(self.policy, agent, noise_source, self.strategy_handler, self.worker_id,
                             sigma=noise_std, random_seed=random_seed, collect_zeta=collect_zeta,
                             eval_prob=eval_prob, eval_only=self.eval_only)

def run(eval_only, render=False, server_port=None, server_address="localhost"):
    runner = ClientRunner(eval_only=eval_only, render=render, server_port=server_port, server_address=server_address)
    runner.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action="store_true", help="Run in eval only mode.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    parser.add_argument("--server_port", type=int, default=None, help="Port of the server. (default: %(default)s)")
    parser.add_argument("--server_address", type=str, default="localhost", help="Address of the server. (default: %(default)s)")
    parser.add_argument("-j", "--num_procs", type=int, default=1, help="Number of processes to use. (default: %(default)s)")

    args = parser.parse_args()
    eval_only = args.eval_only
    render = args.render

    if args.num_procs != 1 and render:
        raise ValueError("Cannot render in multi-process mode.")
    
    if eval_only:
        print("Running in eval only mode")
    if render:
        print("Rendering enabled")

    if args.num_procs > 1:
        procs = [
            Process(
                target=run,
                args=(
                    eval_only,
                    render,
                    args.server_port,
                    args.server_address
            )
            ) for _ in range(args.num_procs)
        ]

        for p in procs:
            p.start()
        
        for p in procs:
            p.join()

    else:
        run(eval_only, render, args.server_port, args.server_address)
    
