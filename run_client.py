import time
from custom_envs import simple_trap_env
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

torch.set_num_threads(1)


class ClientRunner(object):
    def __init__(self):
        self.policy_reward = 0
        self.policy_entropy = 0
        self.policy_novelty = 0
        self.worker = None
        self.strategy_handler = None
        self.policy = None
        self.env = None
        self.rng = None

        self.client = RPCClient()

    @torch.no_grad()
    def run(self):
        client = self.client
        client.connect(address="localhost", port=1025)
        self.receive_config()

        policy = self.policy
        worker = self.worker

        policy.deserialize(client.current_state.policy_params)
        strategy_handler = self.strategy_handler
        strategy_handler.add_policy(policy)
        worker.update(client.current_state)
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
            # Don't use elif here. If the connection falls and is re-established, new_experiment_flag may be set.
            if status == RPCClient.NEW_EXPERIMENT_FLAG:
                print("CLIENT GOT NEW EXPERIMENT")
                cfg = client.current_state.cfg
                self.configure(cfg["env_id"], cfg["normalize_obs"], cfg["obs_stats_update_chance"], cfg["noise_std"],
                               cfg["random_seed"], cfg["eval_prob"], cfg["max_strategy_history_size"])

                strategy_handler = self.strategy_handler
                policy = self.policy
                worker = self.worker

                policy.deserialize(client.current_state.policy_params)
                strategy_handler.add_policy(policy)
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
                       cfg["random_seed"], cfg["eval_prob"], cfg["max_strategy_history_size"])

    def configure(self, env_id="Walker2d-v2", normalize_obs=True, obs_stats_update_chance=0.01,
                  noise_std=0.02, random_seed=124, eval_prob=0.05, max_strategy_history_size=200):

        print("Env: {}\nSeed: {}\nNoise std: {}\nNormalize obs: {}".format(env_id, random_seed, noise_std, normalize_obs))
        random_seed = int(random_seed)
        self.rng = np.random.RandomState(random_seed)

        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.env, self.policy, strategy_distance_fn = init_helper.get_init_data(env_id, random_seed)

        noise_source = RNGNoiseSource(self.policy.num_params, random_seed=random_seed)

        self.strategy_handler = StrategyHandler(self.policy, strategy_distance_fn,
                                                max_history_size=max_strategy_history_size)

        self.worker = Worker(self.policy, Agent(self.policy, self.env, random_seed, normalize_obs=normalize_obs,
                                                obs_stats_update_chance=obs_stats_update_chance),
                             noise_source, self.strategy_handler, sigma=noise_std, random_seed=random_seed,
                             eval_prob=eval_prob)


if __name__ == "__main__":
    runner = ClientRunner()
    runner.run()
