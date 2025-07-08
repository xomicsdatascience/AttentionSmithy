import torch
from torch import nn
from attention_smithy.utils import repeat_module_consecutively
from attention_smithy.components import EncoderLayer

class Encoder(nn.Module):
    """
    A full encoder as described in the Attention is All You Need paper.
        An encoder consists of several encoder layers. This class facilitates
        running data through each encoder layer.

    """
    def __init__(self,
                 layer: EncoderLayer,
                 number_of_layers: int,
                 ) -> None:
        """
        Args:
            layer (EncoderLayer): An instance of the EncoderLayer class.
                This layer will be duplicated across the encoder.
            number_of_layers (int): The number of times the given layer
                should be duplicated throughout the encoder.
        """
        super().__init__()
        self.layers = repeat_module_consecutively(layer, number_of_layers)
        self.norm = nn.LayerNorm(layer.embedding_dimension)

    def forward(self,
                src: torch.Tensor,
                src_padding_mask: torch.Tensor,
                **kwargs
                ) -> torch.Tensor:
        """
        See args and return value of EncoderLayer forward function
        """
        for layer in self.layers:
            src = layer(src, src_padding_mask, **kwargs)
        return self.norm(src)

    def freeze_layers(self, number_of_layers):
        """
        Args:
            number_of_layers (int): The number of layers to be frozen,
                starting with layer index 0.
        """
        for idx, module in enumerate(self.layers):
            if idx < number_of_layers:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

