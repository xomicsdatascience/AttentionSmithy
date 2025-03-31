import torch
from torch import nn
from copy import deepcopy
from attention_smithy.utils import repeat_module_consecutively
from attention_smithy.components import PerceiverEncoderLayer

class PerceiverEncoder(nn.Module):
    """
    A Perceiver encoder that initializes the latent space and applies multiple
    Perceiver encoder layers. Weight sharing across layers can be enabled, which
    reuses a single layer in an RNN-like fashion.
    """
    def __init__(self,
                 latent_dim: int,
                 latent_length: int,
                 perceiver_encoder_layer: PerceiverEncoderLayer,
                 number_of_layers: int,
                 shared_weights: bool = False,
                 ) -> None:
        """
        Args:
            latent_dim (int): The size of the latent space embedding.
            latent_length (int): The number of latent tokens.
            perceiver_encoder_layer (PerceiverEncoderLayer): An instance of the PerceiverEncoderLayer class.
            number_of_layers (int): The number of times the encoder layer is applied.
            shared_weights (bool, optional): If True, a single encoder layer is reused
                across all iterations. Defaults to False.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_length = latent_length
        self.number_of_layers = number_of_layers
        self.shared_weights = shared_weights
        self.latents = nn.Parameter(torch.randn(1, latent_length, latent_dim))
        if shared_weights:
            self.layer = perceiver_encoder_layer
        else:
            self.layers = repeat_module_consecutively(perceiver_encoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self,
                src: torch.Tensor,
                src_padding_mask: torch.Tensor,
                **kwargs
                ) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): The tokenized input of shape
                (batch_size, sequence_length, input_dim).
            src_padding_mask (torch.Tensor): The padding mask for the input tokens.
            kwargs: Additional parameters for custom components.
        Returns:
            torch.Tensor: The processed latent representation of shape
                (batch_size, latent_length, latent_dim).
        """
        batch_size = src.size(0)
        latent = self.latents.expand(batch_size, -1, -1)
        if self.shared_weights:
            for _ in range(self.number_of_layers):
                latent = self.layer(latent, src, src_padding_mask, **kwargs)
        else:
            for layer in self.layers:
                latent = layer(latent, src, src_padding_mask, **kwargs)
        return self.norm(latent)