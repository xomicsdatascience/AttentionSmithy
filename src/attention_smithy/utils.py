import random
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy

def repeat_module_consecutively(
        module: nn.Module,
        number_of_repeats: int,
) -> nn.ModuleList:
    """
    Args:
        module (nn.Module): A pytorch module that the user wants to repeat consecutively.
            NOTE: The input and output shapes for this module must be identical, as one
            repeat will feed into another.
        number_of_repeats (int): The number of consecutive repeats of the module.
    Returns:
        modules (nn.ModuleList): A list of modules that are repeats of the given module.
    """
    modules = nn.ModuleList([copy.deepcopy(module) for _ in range(number_of_repeats)])
    _reset_all_copied_initial_weights(modules)
    return modules

def _reset_all_copied_initial_weights(
        modules: nn.ModuleList,
) -> nn.ModuleList:
    for _, submodule in modules.named_modules():
        if hasattr(submodule, 'reset_parameters'):
            submodule.reset_parameters()

def seed_everything(
        seed: int,
) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_causal_mask(
        sequence_length: int,
        device: torch.device = None
) -> torch.Tensor:
    """
    Creates a boolean tensor where values above the diagonal are False, all others
        are True. This creates a "causal" mask in attention that prevents tokens
        from receiving information from future tokens. This is used in things like
        GPT to optimize training. This is applied to attention_scores before softmax.
    NOTE: Causal masking only applies to self-attention, so sequence_length represents
        both query and key sequence lengths.

    Args:
        sequence_length (int): The dimensions of the square mask.
        device (torch.device, optional): Device on which to create the mask.

    Returns:
        subsequent_mask (torch.Tensor): A causal mask for attention scores, of shape
            (1, sequence_length, sequence_length).
    """
    attn_shape = (1, sequence_length, sequence_length)
    subsequent_mask = torch.triu(torch.ones(attn_shape, device=device), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class ReGLU(nn.Module):
    """
    ReGLU: Gated Linear Unit variant that uses ReLU for gating.
    Splits the last dimension into two halves, applies ReLU to the first,
    and multiplies elementwise with the second half.
    """
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.relu(a) * b

class GEGLU(nn.Module):
    """
    GEGLU: Gated Linear Unit variant that uses GELU for gating.
    Splits the last dimension into two halves, applies GELU to the first,
    and multiplies elementwise with the second half.
    """
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.gelu(a) * b

class Squareplus(nn.Module):
    """
    Squareplus activation function.
    Defined as: (x + sqrt(x^2 + b)) / 2.
    The parameter `b` can be adjusted; default is 4.0.
    """
    def __init__(self, b: float = 4.0):
        super().__init__()
        self.b = b

    def forward(self, x):
        return 0.5 * (x + torch.sqrt(x**2 + self.b))

class EATLU(nn.Module):
    """
    Expanded ArcTan Linear Unit (EATLU).
    Applies arctan to the input scaled by a learnable parameter alpha.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return x * torch.arctan(self.alpha * x)

def select_activation_function_module(activation_param: str, **kwargs) -> nn.Module:
    if activation_param == "relu":
        return nn.ReLU()
    elif activation_param == "leaky_relu_steep":
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation_param == "leaky_relu_slight":
        return nn.LeakyReLU(negative_slope=0.01)
    elif activation_param == "sigmoid":
        return nn.Sigmoid()
    elif activation_param == "tanh":
        return nn.Tanh()
    elif activation_param == "rrelu":
        return nn.RReLU()
    elif activation_param == "selu":
        return nn.SELU()
    elif activation_param == "gelu":
        return nn.GELU()
    elif activation_param == "silu" or activation_param == "swish":
        return nn.SiLU()
    elif activation_param == "mish":
        return nn.Mish()
    elif activation_param == "reglu":
        return ReGLU()
    elif activation_param == "geglu":
        return GEGLU()
    elif activation_param == "squareplus":
        return Squareplus(**kwargs)
    elif activation_param == "eatlu":
        return EATLU(**kwargs)
    else:
        raise ValueError(f"Unsupported activation function: {activation_param}")

def get_available_gpu_count():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        return 1
    elif torch.backends.opencl.is_available():
        return torch.opencl.device_count()
    else:
        return 0
