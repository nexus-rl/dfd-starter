import numpy as np
import torch
from policies import MujocoPolicy


class StrategyPoint(object):
    def __init__(self, policy, flat):
        self.flat = flat.copy()
        self.policy = policy
        self.strategy = None

        self.closest = None
        self.second_closest = None
        self.dists = {}
        self.reset_dists()

    @torch.no_grad()
    def evaluate_strategy(self, zeta):
        old = self.policy.get_trainable_flat()
        self.policy.set_trainable_flat(self.flat)

        self.strategy = self.policy.get_strategy(zeta)
        self.policy.set_trainable_flat(old)

        return self.strategy

    def add_dist(self, key, dist):
        if dist < self.closest[1]:
            self.second_closest = self.closest[:]
            self.closest[0] = key
            self.closest[1] = dist

        elif dist < self.second_closest[1] and key != self.closest[0]:
            self.second_closest[0] = key
            self.second_closest[1] = dist

    def reset_dists(self):
        self.closest = [None, np.inf]
        self.second_closest = [None, np.inf]
