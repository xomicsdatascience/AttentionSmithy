import torch
from torch import nn
from attention_smithy.components import SublayerUnit, MultiheadAttention, FeedForwardNetwork, Encoder

class PerceiverEncoderLayer(nn.Module):
    """
    A single Perceiver encoder layer, consisting of:
        1. A cross-attention mechanism that maps input tokens to a latent space.
        2. A feed-forward network applied after cross-attention.
        3. Self-attention within the latent space via an encoder.
    """
    def __init__(self,
                 latent_dim: int,
                 cross_attention: MultiheadAttention,
                 feed_forward: FeedForwardNetwork,
                 latent_encoder: Encoder,
                 dropout: float = 0.0,
                 ) -> None:
        """
        Args:
            latent_dim (int): The size of the latent space embedding.
            cross_attention (MultiheadAttention): Multihead attention for
                mapping input tokens to the latent representation.
            feed_forward (FeedForwardNetwork): A feed-forward network applied
                after the cross-attention step.
            latent_encoder (Encoder): Encoder instance that applies self-attention
                within the latent space.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """
        super().__init__()
        self.cross_attention_sublayer = SublayerUnit(cross_attention, latent_dim, dropout)
        self.feed_forward_sublayer = SublayerUnit(feed_forward, latent_dim, dropout)
        self.latent_encoder = latent_encoder

    def forward(self,
                latent: torch.Tensor,
                src: torch.Tensor,
                src_padding_mask: torch.Tensor,
                **kwargs
                ) -> torch.Tensor:
        """
        Args:
            latent (torch.Tensor): The latent space representation tensor of shape
                (batch_size, latent_length, latent_dim).
            src (torch.Tensor): The tokenized input tensor of shape
                (batch_size, sequence_length, input_dim).
            src_padding_mask (torch.Tensor): The padding mask for input tokens,
                shape (batch_size, sequence_length).
            kwargs: Additional parameters for custom components.

        Returns:
            torch.Tensor: Updated latent representation of shape
                (batch_size, latent_length, latent_dim).
        """
        latent = self.cross_attention_sublayer(
            latent,
            input_key=src,
            input_value=src,
            padding_and_loss_attention_mask=src_padding_mask,
            **kwargs
        )
        latent = self.feed_forward_sublayer(latent)
        return self.latent_encoder(latent, **kwargs)