import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    """
    LARS (Layer-wise Adaptive Rate Scaling) optimizer.
    Standard optimizer for linear probing on frozen self-supervised / foundation ViT backbones (MoCo v3, MAE, DOFA).
    """
    def __init__(self, params, lr=1.0, momentum=0.9, weight_decay=0.0, trust_coefficient=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coefficient=trust_coefficient, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["trust_coefficient"]
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if p.ndim > 1:  # Weight matrices (not 1D biases or batchnorm scales)
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(d_p)

                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)

                    if w_norm > 0 and g_norm > 0:
                        local_lr = eta * w_norm / (g_norm + weight_decay * w_norm + eps)
                    else:
                        local_lr = 1.0
                    actual_lr = lr * local_lr
                else:
                    actual_lr = lr

                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p)

                p.add_(buf, alpha=-actual_lr)

        return loss
