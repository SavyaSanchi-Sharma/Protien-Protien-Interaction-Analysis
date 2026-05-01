"""Lion optimizer (Chen et al. 2023, https://arxiv.org/abs/2302.06675).

Why Lion for the 15:85 positive:negative residue split: every parameter
gets an update of magnitude exactly `lr` (sign step), so the majority
class can't dominate the direction the way it does under AdamW's
per-parameter adaptive scaling.

Tuning vs AdamW: lr ~ 1/10, weight_decay ~ 10-100x, betas (0.9, 0.99).
Works with AMP `GradScaler.unscale_(opt)`.
"""

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                # Clone exp_avg before in-place ops so the running EMA used in
                # the next step isn't corrupted.
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1)
                update.sign_()
                p.data.add_(update, alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
