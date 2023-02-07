from torch.optim import Optimizer
import torch
import numpy as np
from utils import math_helpers


class DSGD(Optimizer):
    def __init__(self, params, lr, min_scale=0.23, max_scale=1.0):
        super().__init__(params, {"lr": lr})
        self.lr = lr
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.coef = 1
        self.lr_scale = 1
        self.steps = 0
        self._compute_coef()

    @torch.no_grad()
    def step(self, closure=None):
        flat_grad = torch.FloatTensor()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    flat_grad = torch.cat((flat_grad, p.grad.view(-1)))

        norm = flat_grad.norm().item()
        assert norm > 0, "DSGD ENCOUNTERED GRADIENT WITH NORM OF ZERO"

        coef = self.lr * self.coef * self.lr_scale / norm
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                step = p.numel()
                grad_slice = flat_grad[idx:idx+step].view_as(p)
                p.sub_(coef*grad_slice)
                idx += step

        self.steps += 1

    def adjust_lr(self, omega):
        self.lr_scale = math_helpers.affine_transform(omega.omega, omega.min_omega,
                                                      omega.max_omega, self.min_scale,
                                                      self.max_scale)

    def _compute_coef(self):
        d = 0
        for group in self.param_groups:
            for p in group["params"]:
                d += p.numel()
        self.coef = np.sqrt(d)
