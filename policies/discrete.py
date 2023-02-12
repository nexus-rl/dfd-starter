from torch.distributions import Categorical
import torch.nn as nn
import torch
import numpy as np
from policies import Policy


class DiscretePolicy(Policy):
    def __init__(self, n_inputs, n_actions, seed=124):
        super().__init__(n_inputs, n_actions, seed=seed)
        self._build_model()
        self.eval()
        self.num_params = len(self.get_trainable_flat())
        self._init_params()

    def get_action(self, x, deterministic=False):
        probs = self.forward(x)
        if deterministic:
            return probs.argmax().item()

        distr = Categorical(probs=probs)
        act = distr.sample()

        return act.item()

    def get_entropy(self, x):
        probs = self.forward(x).detach()
        distr = Categorical(probs=probs)
        return distr.entropy().mean().item()

    def get_strategy(self, x):
        return self.forward(x).numpy()

    def _build_model(self):
        h1 = 128
        h2 = 128
        h3 = 128
        h4 = 128
        self.model = nn.Sequential(
            nn.BatchNorm1d(self.input_shape),
            nn.Linear(self.input_shape, h1),
            nn.ReLU(),

            nn.BatchNorm1d(h1),
            nn.Linear(h1, h2),
            nn.ReLU(),

            nn.BatchNorm1d(h2),
            nn.Linear(h2, h3),
            nn.ReLU(),

            nn.BatchNorm1d(h3),
            nn.Linear(h3, h4),
            nn.ReLU(),

            nn.BatchNorm1d(h4),
            nn.Linear(h4, self.output_shape),
            nn.Softmax(dim=-1))
