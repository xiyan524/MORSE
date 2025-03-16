import torch
import math
from torch import nn

def swish(x):
    return x * torch.sigmoid(x)


def gelu(x):
    """
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "mish": mish}
