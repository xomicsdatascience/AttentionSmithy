import torch
import pytest
import torch.nn as nn

from attention_smithy.components import PerceiverEncoderLayer, PerceiverEncoder

# Dummy modules to simulate behavior
class DummyCrossAttention(nn.Module):
    def forward(self, query, input_key, input_value, padding_and_loss_attention_mask, **kwargs):
        # Simulate cross-attention by simply adding 1 to the query.
        return query + 1.0

class DummyFeedForward(nn.Module):
    def forward(self, x):
        # Simulate feed-forward by adding 2.
        return x + 2.0

class DummyEncoder(nn.Module):
    def forward(self, latent, **kwargs):
        # Simulate latent self-attention by adding 3.
        return latent + 3.0

class DummyNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# ---------------------------------------------------------------------------
# PerceiverEncoderLayer Tests
# ---------------------------------------------------------------------------
def test__PerceiverEncoderLayer__forward_output_shape_and_values():
    batch_size = 2
    latent_length = 4
    latent_dim = 8
    src_length = 5
    input_dim = 10

    latent = torch.zeros(batch_size, latent_length, latent_dim)
    src = torch.zeros(batch_size, src_length, input_dim)
    src_padding_mask = torch.zeros(batch_size, src_length, dtype=torch.bool)

    cross_attention = DummyCrossAttention()
    feed_forward = DummyFeedForward()
    latent_encoder = DummyEncoder()

    perceiver_layer = PerceiverEncoderLayer(
        latent_dim=latent_dim,
        cross_attention=cross_attention,
        feed_forward=feed_forward,
        latent_encoder=latent_encoder,
        dropout=0.0,
    )
    perceiver_layer.cross_attention_sublayer.post_norm = DummyNorm()
    perceiver_layer.feed_forward_sublayer.post_norm = DummyNorm()

    output = perceiver_layer(latent, src, src_padding_mask)
    # Expected transformation:
    # cross_attention: latent + 1
    # feed_forward: (latent + 1) + 2 = latent + 3
    # latent_encoder: latent + 3 = latent + 6
    expected = latent + 7.0

    assert output.shape == (batch_size, latent_length, latent_dim)
    assert torch.allclose(output, expected), "PerceiverEncoderLayer forward did not produce expected output."

# ---------------------------------------------------------------------------
# PerceiverEncoder Tests (Shared Weights)
# ---------------------------------------------------------------------------
def test__PerceiverEncoder__shared_weights_forward_output_shape():
    batch_size = 2
    src_length = 5
    input_dim = 10
    latent_length = 4
    latent_dim = 8
    number_of_layers = 3

    src = torch.zeros(batch_size, src_length, input_dim)
    src_padding_mask = torch.zeros(batch_size, src_length, dtype=torch.bool)

    cross_attention = DummyCrossAttention()
    feed_forward = DummyFeedForward()
    latent_encoder = DummyEncoder()
    dummy_layer = PerceiverEncoderLayer(
        latent_dim=latent_dim,
        cross_attention=cross_attention,
        feed_forward=feed_forward,
        latent_encoder=latent_encoder,
        dropout=0.0
    )

    encoder = PerceiverEncoder(
        latent_dim=latent_dim,
        latent_length=latent_length,
        perceiver_encoder_layer=dummy_layer,
        number_of_layers=number_of_layers,
        shared_weights=True,
    )

    # For deterministic behavior, initialize latent tokens to zeros.
    with torch.no_grad():
        encoder.latents.copy_(torch.zeros(1, latent_length, latent_dim))

    output = encoder(src, src_padding_mask)
    # Due to LayerNorm at the end, we verify the output shape.
    assert output.shape == (batch_size, latent_length, latent_dim)

# ---------------------------------------------------------------------------
# PerceiverEncoder Tests (Unique Layers)
# ---------------------------------------------------------------------------
def test__PerceiverEncoder__unique_layers_forward_output_shape():
    batch_size = 2
    src_length = 5
    input_dim = 10
    latent_length = 4
    latent_dim = 8
    number_of_layers = 3

    src = torch.zeros(batch_size, src_length, input_dim)
    src_padding_mask = torch.zeros(batch_size, src_length, dtype=torch.bool)

    cross_attention = DummyCrossAttention()
    feed_forward = DummyFeedForward()
    latent_encoder = DummyEncoder()
    dummy_layer = PerceiverEncoderLayer(
        latent_dim=latent_dim,
        cross_attention=cross_attention,
        feed_forward=feed_forward,
        latent_encoder=latent_encoder,
        dropout=0.0
    )

    encoder = PerceiverEncoder(
        latent_dim=latent_dim,
        latent_length=latent_length,
        perceiver_encoder_layer=dummy_layer,
        number_of_layers=number_of_layers,
        shared_weights=False,
    )
    with torch.no_grad():
        encoder.latents.copy_(torch.zeros(1, latent_length, latent_dim))
    output = encoder(src, src_padding_mask)
    assert output.shape == (batch_size, latent_length, latent_dim)