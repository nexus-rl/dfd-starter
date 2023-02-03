from torch.distributions import Categorical
import torch.nn as nn
import torch
from policies import Policy


class AtariPolicy(Policy):
    def __init__(self, n_inputs, n_actions, seed=124):
        super().__init__(n_inputs, n_actions, seed=seed)
        in_channels = 4
        self.input_shape = (in_channels, n_inputs[0], n_inputs[1])
        self._build_model()
        self.model.eval()
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
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.input_shape[0], out_channels=16, kernel_size=[8, 8], stride=[4, 4]),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[4, 4], stride=[2, 2]),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=2592, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=self.output_shape),
            nn.Softmax(dim=-1)
        )
