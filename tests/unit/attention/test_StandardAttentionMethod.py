import torch
from attention_smithy.attention import StandardAttentionMethod

def test__StandardAttentionMethod__forward_pass_functions_as_expected():
    q = torch.tensor([[[[1e-1, 2e-1, 3e-1], [4e-1, 5e-1, 6e-1]]]])
    k = torch.tensor([[[[1e-1, 2e-1, 3e-1], [4e-1, 5e-1, 6e-1]]]])
    v = torch.tensor([[[[1e-1, 2e-1, 3e-1], [4e-1, 5e-1, 6e-1]]]])
    attention = StandardAttentionMethod()
    output, attn_probs = attention(q, k, v)

    expected_attn_probs = torch.tensor([[[
        [0.4551, 0.5449],
        [0.3894, 0.6106]
    ]]])
    expected_output = torch.tensor([[[
        [0.2635, 0.3635, 0.4635],
        [0.2832, 0.3832, 0.4832]
    ]]])

    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)
