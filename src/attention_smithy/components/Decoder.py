import torch
from torch import nn
from attention_smithy.utils import repeat_module_consecutively
from attention_smithy.components import DecoderLayer

class Decoder(nn.Module):
    """
    A full decoder as described in the Attention is All You Need paper.
        A decoder consists of several decoder layers. This class facilitates
        running data through each decoder layer.

    """
    def __init__(self,
                 layer: DecoderLayer,
                 number_of_layers: int,
                 ):
        """
        Args:
            layer (DecoderLayer): An instance of the DecoderLayer class.
                This layer will be duplicated across the decoder.
            number_of_layers (int): The number of times the given layer
                should be duplicated throughout the decoder.
        """
        super().__init__()
        self.layers = repeat_module_consecutively(layer, number_of_layers)
        self.norm = nn.LayerNorm(layer.embedding_dimension)

    def forward(self,
                tgt: torch.Tensor,
                src: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                src_padding_mask: torch.Tensor,
                **kwargs
                ):
        """
        See args and return value of DecoderLayer forward function
        """
        for layer in self.layers:
            tgt = layer(tgt, src, tgt_padding_mask, src_padding_mask, **kwargs)
        return self.norm(tgt)
