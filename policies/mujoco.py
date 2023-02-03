from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from policies import Policy
from utils import torch_helpers


class MujocoPolicy(Policy):
    def __init__(self, n_inputs, n_actions, seed=124):
        super().__init__(n_inputs, n_actions, seed=seed)
        self._build_model()
        self.num_params = len(self.get_trainable_flat())
        self._init_params()

    def get_action(self, x, deterministic=False):
        mean, std = self.forward(x)
        if deterministic:
            return mean.flatten().tolist()

        distr = Normal(mean, std)
        act = distr.sample()
        return act.flatten().tolist()

    def get_entropy(self, x):
        mean, std = self.forward(x)
        distr = Normal(mean, std)
        return distr.entropy().sum(dim=-1).mean().item()

    def get_strategy(self, x):
        return np.concatenate(self.forward(x), axis=-1)

    def _build_model(self):
        h1 = 64
        h2 = 64
        self.model = nn.Sequential(
            nn.Linear(self.input_shape, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, self.output_shape * 2),
            torch_helpers.MapContinuousToAction())
