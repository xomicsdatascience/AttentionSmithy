import pytest
import torch
from attention_smithy.attention import LongformerAttentionMethod
from attention_smithy.attention.tests import ManuallyMaskedAttentionMethod
from attention_smithy.numeric_embeddings import NumericEmbeddingManager

@pytest.fixture
def numeric_embedding_manager():
    return NumericEmbeddingManager([])

def test__LongformerAttentionMethod__manual_mask_8x8__window_width_2(numeric_embedding_manager):
    torch.manual_seed(0)

    batch_size = 12
    num_heads = 6
    seq_len = 8
    head_dim = 4
    window = 2

    q = torch.rand(batch_size, num_heads, seq_len, head_dim)
    k = q.clone()
    v = q.clone()

    manual_mask = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ], dtype=torch.float32)

    # Run manually masked attention (expected baseline)
    manual_attention = ManuallyMaskedAttentionMethod()
    expected_output, expected_probs = manual_attention(
        q, k, v,
        numeric_embedding_manager=numeric_embedding_manager,
        manual_attention_mask=manual_mask
    )

    # Run Longformer attention
    longformer_attention = LongformerAttentionMethod(attention_window=window)
    output, attn_probs = longformer_attention(
        q, k, v,
        numeric_embedding_manager=numeric_embedding_manager,
    )
    assert torch.allclose(output, expected_output, atol=1e-4), "Attention outputs do not match"

def test__LongformerAttentionMethod__global_tokens_0_and_2__window_width_2(numeric_embedding_manager):
    torch.manual_seed(0)

    batch_size = 12
    num_heads = 6
    seq_len = 8
    head_dim = 4
    window = 2

    q = torch.rand(batch_size, num_heads, seq_len, head_dim)
    k = q.clone()
    v = q.clone()

    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.int)
    attention_mask[:, 0] = 1
    attention_mask[:, 2] = 1

    # --- Expected attention mask matrix (based on your definition above)
    manual_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],  # global (token 0)
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],  # global (token 2)
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 1, 1],
    ], dtype=torch.float32)

    # Expected output using manually masked standard attention
    manual_attention = ManuallyMaskedAttentionMethod()
    expected_output, expected_probs = manual_attention(
        q, k, v,
        numeric_embedding_manager=numeric_embedding_manager,
        manual_attention_mask=manual_mask
    )

    # Actual output using Longformer attention with global tokens
    longformer_attention = LongformerAttentionMethod(attention_window=window)
    output, attn_probs = longformer_attention(
        q, k, v,
        numeric_embedding_manager=numeric_embedding_manager,
        attention_mask=attention_mask,
    )
    assert torch.allclose(output, expected_output, atol=1e-4), "Global attention outputs do not match"

@pytest.mark.skip(reason="using expected numbers to simplify the output for debugging purposes.")
def test__LongformerAttentionMethod__manual_mask_8x8_window2__simplifed_printing_example(numeric_embedding_manager):
    torch.manual_seed(0)

    batch_size = 1
    num_heads = 1
    seq_len = 8
    head_dim = 4
    window = 2

    q = torch.ones(batch_size, num_heads, seq_len, head_dim)

    # k: simple identity-like encoding (distinct per token)
    k = torch.zeros(batch_size, num_heads, seq_len, head_dim)
    for i in range(seq_len):
        k[0, 0, i] = torch.tensor([i + 1, i + 2, i + 3, i + 4], dtype=torch.float32)

    # v: now each token has a different vector across head dimensions
    v = torch.zeros(batch_size, num_heads, seq_len, head_dim)
    for i in range(seq_len):
        v[0, 0, i] = torch.tensor([i + 0.1, i + 0.2, i + 0.3, i + 0.4], dtype=torch.float32)

    manual_mask = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ], dtype=torch.float32)

    # Run manually masked attention (expected baseline)
    manual_attention = ManuallyMaskedAttentionMethod()
    expected_output, expected_probs = manual_attention(
        q, k, v,
        numeric_embedding_manager=numeric_embedding_manager,
        manual_attention_mask=manual_mask
    )

    # Run Longformer attention
    longformer_attention = LongformerAttentionMethod(attention_window=window)
    output, attn_probs = longformer_attention(
        q, k, v,
        numeric_embedding_manager=numeric_embedding_manager,
    )
    print('\n'*10)
    print(output)
    print(expected_output)

    assert torch.allclose(output, expected_output, atol=1e-4), "Attention outputs do not match"

@pytest.mark.skip(reason="Cannot currently shut off local token attention - this test assumes no local attention.")
def test__LongformerAttentionMethod__global_tokens_0_and_2(numeric_embedding_manager):
    torch.manual_seed(0)

    batch_size = 1
    num_heads = 1
    seq_len = 8
    head_dim = 4
    window = 2

    q = torch.rand(batch_size, num_heads, seq_len, head_dim)
    k = q.clone()
    v = q.clone()

    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.int)
    attention_mask[:, 0] = 1
    attention_mask[:, 2] = 1

    # --- Expected attention mask matrix (based on your definition above)
    manual_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],  # global (token 0)
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],  # global (token 2)
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
    ], dtype=torch.float32)

    # Expected output using manually masked standard attention
    manual_attention = ManuallyMaskedAttentionMethod()
    expected_output, expected_probs = manual_attention(
        q, k, v,
        numeric_embedding_manager=numeric_embedding_manager,
        manual_attention_mask=manual_mask
    )

    # Actual output using Longformer attention with global tokens
    longformer_attention = LongformerAttentionMethod(attention_window=window)
    output, attn_probs = longformer_attention(
        q, k, v,
        numeric_embedding_manager=numeric_embedding_manager,
        attention_mask=attention_mask,
    )

    print('\n')
    print(output)
    print(expected_output)
    assert torch.allclose(output, expected_output, atol=1e-4), "Global attention outputs do not match"
