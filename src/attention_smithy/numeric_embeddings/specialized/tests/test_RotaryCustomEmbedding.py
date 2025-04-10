import torch
import pytest
from attention_smithy.numeric_embeddings import RotaryCustomEmbedding

# --------------- FIXTURE -----------------

@pytest.fixture
def dummy_tensor():
    batch, num_heads, seq_len, dim = 2, 4, 6, 64
    return torch.randn(batch, num_heads, seq_len, dim)

@pytest.fixture
def custom_values(dummy_tensor):
    batch = dummy_tensor.shape[0]
    seq_len = dummy_tensor.shape[-2]
    values = torch.tensor([0.4, 1.2, 5.5, 13.3, 21.0, 30.5], dtype=torch.float32)
    return values.unsqueeze(0).expand(batch, -1)


@pytest.fixture
def custom_rotary():
    return RotaryCustomEmbedding(head_dimension=64)

# --------------- TESTS -------------------

def test__RotaryCustomEmbedding__preserves_shape(dummy_tensor, custom_values, custom_rotary):
    output = custom_rotary(dummy_tensor, rotary_custom_values=custom_values)
    assert output.shape == dummy_tensor.shape

def test__RotaryCustomEmbedding__fails_on_mismatched_positions(dummy_tensor, custom_rotary):
    batch = dummy_tensor.shape[0]
    wrong_positions = torch.tensor([[0.1, 0.2]] * batch, dtype=torch.float32)
    with pytest.raises(ValueError, match="Expected rotary_custom_values with sequence length .* but got shape"):
        custom_rotary(dummy_tensor, rotary_custom_values=wrong_positions)

def test__RotaryCustomEmbedding__different_positions_give_different_results(dummy_tensor, custom_rotary):
    batch, seq_len = dummy_tensor.shape[0], dummy_tensor.shape[-2]
    positions_1 = torch.linspace(0.0, 1.0, steps=seq_len, dtype=torch.float32).unsqueeze(0).expand(batch, -1)
    positions_2 = torch.linspace(10.0, 20.0, steps=seq_len, dtype=torch.float32).unsqueeze(0).expand(batch, -1)

    out1 = custom_rotary(dummy_tensor, rotary_custom_values=positions_1)
    out2 = custom_rotary(dummy_tensor, rotary_custom_values=positions_2)

    assert not torch.allclose(out1, out2, atol=1e-5), "Outputs should differ for different custom positions"

def test__RotaryCustomEmbedding__no_nan_or_inf(dummy_tensor, custom_values, custom_rotary):
    output = custom_rotary(dummy_tensor, rotary_custom_values=custom_values)
    assert not torch.isnan(output).any(), "Output contains NaNs"
    assert not torch.isinf(output).any(), "Output contains Infs"