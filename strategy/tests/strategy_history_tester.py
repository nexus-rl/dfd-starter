import time

from policies import Policy
from strategy import SparseHistoryManager
import numpy as np
import torch.nn as nn


class DummyPolicy(Policy):
    def __init__(self, n_inputs, n_outputs, rng_seed=1):
        super().__init__(n_inputs, n_outputs)
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)
        h = 256
        self.model = nn.Sequential(nn.Linear(n_inputs, h),
                                   nn.ReLU(),
                                   nn.Linear(h, h),
                                   nn.ReLU(),
                                   nn.Linear(h, h),
                                   nn.ReLU(),
                                   nn.Linear(h, h),
                                   nn.ReLU(),
                                   nn.Linear(h, n_outputs),
                                   nn.ReLU(),
                                   )

    # def forward(self, x):
    #     return np.multiply(x, self.rng_seed)

    def get_action(self, x, deterministic=False):
        return 0

    def get_entropy(self, x):
        return 1

    # def set_trainable_flat(self, flat):
    #     self.rng_seed = flat
    #
    # def get_trainable_flat(self):
    #     return np.asarray(self.rng_seed)


def run_test():
    zeta_size = 200
    history_size = 200
    n_inputs = 100
    n_outputs = 18

    zeta = np.asarray([np.random.randn(n_inputs) for i in range(zeta_size)])
    # zeta = np.asarray([(i, i) for i in range(zeta_size)])
    policy = DummyPolicy(n_inputs, n_outputs, 0)
    handler = SparseHistoryManager(policy, history_size)
    handler.submit_policy(policy)
    handler.evaluate_strategies(zeta)

    initial_policies = [DummyPolicy(n_inputs, n_outputs, i) for i in range(history_size)]
    policies_to_accept = [DummyPolicy(n_inputs, n_outputs, i+history_size) for i in range(10)]
    policies_to_reject = [DummyPolicy(n_inputs, n_outputs, i) for i in range(10)]

    t1 = time.perf_counter()
    for policy in initial_policies:
        handler.submit_policy(policy)
    print("\nFilling time: {:7.4f}".format(time.perf_counter()-t1))

    t1 = time.perf_counter()
    handler.evaluate_strategies(zeta)
    print("\nEval time: {:7.4f}".format(time.perf_counter()-t1))

    t1 = time.perf_counter()
    for policy in policies_to_accept:
        handler.submit_policy(policy)
    print("\nPolicies to accept time: {:7.4f}".format(time.perf_counter() - t1))
    handler.evaluate_strategies(zeta)
    print("\nEval complete")

    for policy in policies_to_reject:
        handler.submit_policy(policy)
    print("\nPolicies rejected")
    t1 = time.perf_counter()
    handler.evaluate_strategies(zeta)
    print("\nEval time: {:7.4f}".format(time.perf_counter() - t1))


if __name__ == "__main__":
    run_test()