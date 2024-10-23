import torch
from attention_smithy.attention import StandardAttentionMethod

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2():
    q = torch.tensor([[[
        [1e-1, 2e-1, 3e-1],
        [4e-1, 5e-1, 6e-1]
    ]]])
    k = torch.tensor([[[
        [1e-1, 2e-1, 3e-1],
        [4e-1, 5e-1, 6e-1]
    ]]])
    v = torch.tensor([[[
        [1e-1, 2e-1, 3e-1],
        [4e-1, 5e-1, 6e-1]
    ]]])
    attention = StandardAttentionMethod()
    output, attn_probs = attention(q, k, v)

    expected_attn_probs = torch.tensor([[[
        [0.4740, 0.5260],
        [0.4354, 0.5646]
    ]]])
    expected_output = torch.tensor([[[
        [0.2578, 0.3578, 0.4578],
        [0.2694, 0.3694, 0.4694]
    ]]])

    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength4():
    q = torch.tensor([[[[1e-1, 2e-1, 3e-1],
                        [4e-1, 5e-1, 6e-1]]]])
    k = torch.tensor([[[[1e-1, 2e-1, 3e-1],
                        [4e-1, 5e-1, 6e-1],
                        [1e-1, 2e-1, 3e-1],
                        [4e-1, 5e-1, 6e-1]]]])
    v = torch.tensor([[[[1e-1, 2e-1, 3e-1],
                        [4e-1, 5e-1, 6e-1],
                        [1e-1, 2e-1, 3e-1],
                        [4e-1, 5e-1, 6e-1]]]])
    attention = StandardAttentionMethod()

    output, attn_probs = attention(q, k, v)

    expected_attn_probs = torch.tensor([[[
        [0.2370, 0.2630, 0.2370, 0.2630],
        [0.2177, 0.2823, 0.2177, 0.2823]
    ]]])
    expected_output = torch.tensor([[[
        [0.2578, 0.3578, 0.4578],
        [0.2694, 0.3694, 0.4694]
    ]]])

    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__apply_causal_masking():
    q = torch.tensor([[[
        [1e-1, 2e-1, 3e-1],
        [4e-1, 5e-1, 6e-1]
    ]]])
    k = torch.tensor([[[
        [1e-1, 2e-1, 3e-1],
        [4e-1, 5e-1, 6e-1]
    ]]])
    v = torch.tensor([[[
        [1e-1, 2e-1, 3e-1],
        [4e-1, 5e-1, 6e-1]
    ]]])
    attention = StandardAttentionMethod(is_causal_masking=True)
    output, attn_probs = attention(q, k, v)

    expected_attn_probs = torch.tensor([[[
        [1.0, 0],
        [0.4354, 0.5646]
    ]]])
    expected_output = torch.tensor([[[
        [1e-1, 2e-1, 3e-1],
        [0.2694, 0.3694, 0.4694]
    ]]])

    assert output.shape == q.shape
    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__batchSize2():
    q = torch.tensor([
        [[
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ]],
        [[
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]]
    ])
    k = torch.tensor([
        [[
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ]],
        [[
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]]
    ])
    v = torch.tensor([
        [[
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ]],
        [[
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]]
    ])
    attention = StandardAttentionMethod()
    output, attn_probs = attention(q, k, v)

    expected_attn_probs = torch.tensor([
        [[
            [0.4740, 0.5260],
            [0.4354, 0.5646]
        ]],
        [[
            [0.4676, 0.5324],
            [0.4290, 0.5710]
        ]],
    ])
    expected_output = torch.tensor([
        [[
            [0.2578, 0.3578, 0.4578],
            [0.2694, 0.3694, 0.4694]
        ]],
        [[
            [0.3097, 0.4097, 0.5097],
            [0.3213, 0.4213, 0.5213]
        ]],
    ])

    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__numHeads2():
    q = torch.tensor([[
        [
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ],
        [
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]
    ]])
    k = torch.tensor([[
        [
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ],
        [
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]
    ]])
    v = torch.tensor([[
        [
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ],
        [
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]
    ]])
    attention = StandardAttentionMethod()
    output, attn_probs = attention(q, k, v)

    expected_attn_probs = torch.tensor([[
        [
            [0.4740, 0.5260],
            [0.4354, 0.5646]
        ],
        [
            [0.4676, 0.5324],
            [0.4290, 0.5710]
        ],
    ]])
    expected_output = torch.tensor([[
        [
            [0.2578, 0.3578, 0.4578],
            [0.2694, 0.3694, 0.4694]
        ],
        [
            [0.3097, 0.4097, 0.5097],
            [0.3213, 0.4213, 0.5213]
        ],
    ]])

    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__numHeads2__apply_causal_masking():
    q = torch.tensor([[
        [
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ],
        [
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]
    ]])
    k = torch.tensor([[
        [
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ],
        [
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]
    ]])
    v = torch.tensor([[
        [
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ],
        [
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]
    ]])
    attention = StandardAttentionMethod(is_causal_masking=True)
    output, attn_probs = attention(q, k, v)

    expected_attn_probs = torch.tensor([[
        [
            [1.0, 0.0],
            [0.4354, 0.5646]
        ],
        [
            [1.0, 0.0],
            [0.4290, 0.5710]
        ],
    ]])
    expected_output = torch.tensor([[
        [
            [1e-1, 2e-1, 3e-1],
            [0.2694, 0.3694, 0.4694]
        ],
        [
            [15e-2, 25e-2, 35e-2],
            [0.3213, 0.4213, 0.5213]
        ],
    ]])

    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__batchSize2__with_padding_and_loss_masking():
    q = torch.tensor([
        [[
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ]],
        [[
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]]
    ])
    k = torch.tensor([
        [[
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ]],
        [[
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]]
    ])
    v = torch.tensor([
        [[
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ]],
        [[
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]]
    ])

    padding_and_loss_attention_mask = torch.tensor([
        [[
            [1, 0],
            [0, 1],
        ]],
        [[
            [1, 1],
            [0, 1],
        ]],
    ])
    attention = StandardAttentionMethod()
    output, attn_probs = attention(q, k, v, padding_and_loss_attention_mask=padding_and_loss_attention_mask)

    expected_attn_probs = torch.tensor([
        [[
            [1.0, 0.0],
            [0.0, 1.0]
        ]],
        [[
            [0.4676, 0.5324],
            [0.0, 1.0]
        ]],
    ])
    expected_output = torch.tensor([
        [[
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ]],
        [[
            [0.3097, 0.4097, 0.5097],
            [45e-2, 55e-2, 65e-2]
        ]],
    ])

    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__numHeads2__padding_and_loss_masking_applies_to_all_heads():
    q = torch.tensor([[
        [
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ],
        [
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]
    ]])
    k = torch.tensor([[
        [
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ],
        [
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]
    ]])
    v = torch.tensor([[
        [
            [1e-1, 2e-1, 3e-1],
            [4e-1, 5e-1, 6e-1]
        ],
        [
            [15e-2, 25e-2, 35e-2],
            [45e-2, 55e-2, 65e-2]
        ]
    ]])
    padding_and_loss_attention_mask = torch.tensor([[
        [
            [1, 1],
            [0, 1],
        ],
    ]])

    attention = StandardAttentionMethod()
    output, attn_probs = attention(q, k, v, padding_and_loss_attention_mask = padding_and_loss_attention_mask)

    expected_attn_probs = torch.tensor([[
        [
            [0.4740, 0.5260],
            [0.0, 1.0]
        ],
        [
            [0.4676, 0.5324],
            [0.0, 1.0]
        ],
    ]])
    expected_output = torch.tensor([[
        [
            [0.2578, 0.3578, 0.4578],
            [4e-1, 5e-1, 6e-1]
        ],
        [
            [0.3097, 0.4097, 0.5097],
            [45e-2, 55e-2, 65e-2]
        ],
    ]])

    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)

