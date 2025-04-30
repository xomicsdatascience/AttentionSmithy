import pytest
import torch
from attention_smithy.attention import LongformerAttentionMethod
from attention_smithy.attention.tests import ManuallyMaskedAttentionMethod
from attention_smithy.numeric_embeddings import NumericEmbeddingManager

@pytest.fixture
def numeric_embedding_manager():
    return NumericEmbeddingManager([])

def test__LongformerAttentionMethod__manual_mask_8x8_window2(numeric_embedding_manager):
    torch.manual_seed(0)

    batch_size = 1
    num_heads = 1
    seq_len = 8
    head_dim = 4
    window = 2

    q = torch.rand(batch_size, num_heads, seq_len, head_dim)
    k = q.clone()
    v = q.clone()

    # --- Manually typed 8x8 sliding window mask (window = 2)
    manual_mask = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0],  # 0 attends to 0,1,2
        [1, 1, 1, 1, 0, 0, 0, 0],  # 1 attends to 0–3
        [1, 1, 1, 1, 1, 0, 0, 0],  # 2 attends to 0–4
        [0, 1, 1, 1, 1, 1, 0, 0],  # 3 attends to 1–5
        [0, 0, 1, 1, 1, 1, 1, 0],  # 4 attends to 2–6
        [0, 0, 0, 1, 1, 1, 1, 1],  # 5 attends to 3–7
        [0, 0, 0, 0, 1, 1, 1, 1],  # 6 attends to 4–7
        [0, 0, 0, 0, 0, 1, 1, 1],  # 7 attends to 5–7
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
        [1, 1, 1, 0, 0, 0, 0, 0],  # 0 attends to 0,1,2
        [1, 1, 1, 1, 0, 0, 0, 0],  # 1 attends to 0–3
        [1, 1, 1, 1, 1, 0, 0, 0],  # 2 attends to 0–4
        [0, 1, 1, 1, 1, 1, 0, 0],  # 3 attends to 1–5
        [0, 0, 1, 1, 1, 1, 1, 0],  # 4 attends to 2–6
        [0, 0, 0, 1, 1, 1, 1, 1],  # 5 attends to 3–7
        [0, 0, 0, 0, 1, 1, 1, 1],  # 6 attends to 4–7
        [0, 0, 0, 0, 0, 1, 1, 1],  # 7 attends to 5–7
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