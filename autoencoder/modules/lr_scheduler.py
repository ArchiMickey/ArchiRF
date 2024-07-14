from torch.optim.lr_scheduler import LambdaLR
from math import sqrt


def InvSqrtDecayLRSched(
    optimizer,
    t_ref = 70000,
    sigma_ref = 0.01
):
    """
    refer to equation 67 and Table1
    """
    def inv_sqrt_decay_fn(t: int):
        return sigma_ref / sqrt(max(t / t_ref, 1.))

    return LambdaLR(optimizer, lr_lambda = inv_sqrt_decay_fn)

def LinearWarmupLRSched(
    optimizer,
    warmup_steps: int
):
    def linear_warmup_fn(t: int):
        return min(1., t / warmup_steps)

    return LambdaLR(optimizer, lr_lambda = linear_warmup_fn)