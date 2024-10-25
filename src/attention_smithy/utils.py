import random
import os
import numpy as np
import torch
from torch import nn
import copy


def clone_module_consecutively(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def create_causal_mask(sequence_length):
    """
    Creates a boolean tensor where values above the diagonal are False, all others
        are True. This creates a "causal" mask in attention that prevents tokens
        from receiving information from future tokens. This is used in things like
        GPT to optimize training. This is applied to attention_scores before softmax.
    NOTE: Causal masking only applies to self-attention, so sequence_length represents
        both query and key sequence lengths.

    Args:
        sequence_length (int): The dimensions of the square mask.

    Returns:
        subsequent_mask (torch.Tensor): A causal mask for attention scores, of shape
            (1, sequence_length, sequence_length).
    """
    attn_shape = (1, sequence_length, sequence_length)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def select_activation_function(activation_param):
    if activation_param == "leaky_relu_steep":
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
