import random
import os
import numpy as np
import torch
from torch import nn
import copy
import inspect
from typing import Optional, Union
from torch import Size, Tensor
import numbers
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init


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

class SiGLU(nn.Module):
    """
    SiGLU: Gated Linear Unit variant that uses Sigmoid for gating.
    Splits the last dimension into two halves, applies Sigmoid to the first,
    and multiplies elementwise with the second half.
    """
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return torch.sigmoid(a) * b

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
    elif activation_param == "siglu":
        return SiGLU()
    elif activation_param == "squareplus":
        return Squareplus(**kwargs)
    elif activation_param == "eatlu":
        return EATLU(**kwargs)
    else:
        raise ValueError(f"Unsupported activation function: {activation_param}")


def select_normalization_module(norm_type: str, **kwargs) -> nn.Module:
    """
    Creates a normalization module based on the provided type.

    Args:
        norm_type (str): A string identifier for the normalization type. Supported values are:
            - "layernorm": Standard Layer Normalization (torch.nn.LayerNorm).
            - "rmsnorm": Root Mean Square Layer Normalization (torch.nn.RMSNorm).
            - None: No normalization; returns a pass-through module (nn.Identity).
        **kwargs: Additional keyword arguments required for module initialization
                  (for example, normalized_shape).

    Returns:
        nn.Module: The corresponding normalization module.
    """
    if norm_type is None or norm_type == "None":
        return nn.Identity()
    norm_type = norm_type.lower()
    if "normalized_shape" not in kwargs:
        kwargs["normalized_shape"] = kwargs["embedding_dimension"]
    if norm_type == "layernorm":
        params = inspect.signature(nn.LayerNorm).parameters
        return nn.LayerNorm(**{k: v for k, v in kwargs.items() if k in params})
    elif norm_type == "rmsnorm":
        params = inspect.signature(RMSNorm).parameters
        return RMSNorm(**{k: v for k, v in kwargs.items() if k in params})
    else:
        raise ValueError("Unsupported normalization type. Supported types are 'layernorm', 'rmsnorm', or None.")


def get_available_gpu_count():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        return 1
    elif torch.backends.opencl.is_available():
        return torch.opencl.device_count()
    else:
        return 0

_shape_t = Union[int, list[int], Size]

class RMSNorm(nn.Module):
    r"""Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/1910.07467.pdf>`__

    .. math::
        y_i = \frac{x_i}{\mathrm{RMS}(x)} * \gamma_i, \quad
        \text{where} \quad \text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}

    The RMS is taken over the last ``D`` dimensions, where ``D``
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the RMS is computed over
    the last 2 dimensions of the input.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: :func:`torch.finfo(x.dtype).eps`
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    """
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: Optional[float]
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs forward pass.
        """
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        """
        Extra information about the module.
        """
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )

