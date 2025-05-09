import torch
from attention_smithy.attention import LinformerAttentionMethod


def test__LinformerAttentionMethod__handles_padding_mask_and_outputs_shapes_correctly():
    batch_size = 2
    num_heads = 4
    seq_len = 8
    head_dim = 16
    embedding_dim = num_heads * head_dim
    projected_k = 4  # compressed size

    torch.manual_seed(0)
    q = torch.rand(batch_size, num_heads, seq_len, head_dim)
    k = q.clone()
    v = q.clone()

    # Padding mask: 1 = valid, 0 = pad
    padding_mask = torch.ones(batch_size, seq_len)
    padding_mask[0, 7] = 0  # pad last token in batch 0
    padding_mask[1, 0] = 0  # pad first token in batch 1

    linformer = LinformerAttentionMethod(
        embedding_dim=embedding_dim,
        sequence_length=seq_len,
        k=projected_k,
        dropout=0.0
    )

    output, approx_probs = linformer(
        q, k, v,
        numeric_embedding_manager=None,
        padding_and_loss_attention_mask=padding_mask
    )

    assert output.shape == (batch_size, num_heads, seq_len, head_dim), f"Unexpected output shape: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaNs"
    assert not torch.isnan(approx_probs).any(), "Approx attention probabilities contain NaNs"