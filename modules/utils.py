from torch.optim.lr_scheduler import LambdaLR
from math import sqrt


def normalize_to_neg1_1(x):
    return x * 2 - 1

def unnormalize_to_0_1(x):
    return (x + 1) * 0.5

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