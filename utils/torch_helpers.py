import torch.nn as nn


def init_weights_orthogonal(module: nn.Module, gain: float = 1) -> None:
    """
    COPIED FROM STABLE-BASELINES3
    Orthogonal initialization (used in PPO and A2C)
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class MapContinuousToAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(x)
        n = x.shape[-1] // 2

        # Map stdev from [-1, 1] to [0.1, 1]
        return x[..., :n], 0.55 + 0.45*x[..., n:]