import pytest
import torch
from attention_smithy.attention import StandardAttentionMethod
from attention_smithy.numeric_embeddings import NumericEmbeddingFacade, ALiBiPositionEmbedding, ALiBiCustomEmbedding

@pytest.fixture
def numeric_embedding_facade():
    return NumericEmbeddingFacade()


def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2(numeric_embedding_facade):
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
    output, attn_probs = attention(
        q, k, v, numeric_embedding_facade=numeric_embedding_facade,
        padding_and_loss_attention_mask=None)

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

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength4(numeric_embedding_facade):
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

    output, attn_probs = attention(q, k, v, numeric_embedding_facade=numeric_embedding_facade,
                                   padding_and_loss_attention_mask=None)

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

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__apply_causal_masking(numeric_embedding_facade):
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
    output, attn_probs = attention(q, k, v, numeric_embedding_facade=numeric_embedding_facade,
                                   padding_and_loss_attention_mask=None)

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

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__batchSize2(numeric_embedding_facade):
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
    output, attn_probs = attention(q, k, v, numeric_embedding_facade=numeric_embedding_facade,
                                   padding_and_loss_attention_mask=None)

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

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__numHeads2(numeric_embedding_facade):
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
    output, attn_probs = attention(q, k, v, numeric_embedding_facade=numeric_embedding_facade,
                                   padding_and_loss_attention_mask=None)

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

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__numHeads2__apply_causal_masking(numeric_embedding_facade):
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
    output, attn_probs = attention(q, k, v, numeric_embedding_facade=numeric_embedding_facade,
                                   padding_and_loss_attention_mask=None)

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

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__alibi_position():
    alibi_position = ALiBiPositionEmbedding(num_heads=1)
    numeric_embedding_facade = NumericEmbeddingFacade(alibi_position=alibi_position)

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
    output, attn_probs = attention(q, k, v, numeric_embedding_facade=numeric_embedding_facade,
                                   padding_and_loss_attention_mask=None)

    expected_attn_probs = torch.tensor([[[
        [0.5977, 0.4023],
        [0.3187, 0.6813]
    ]]])
    expected_output = torch.tensor([[[
        [0.2207, 0.3207, 0.4207],
        [0.3044, 0.4044, 0.5044]
    ]]])
    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__alibi_custom():
    alibi_custom = ALiBiCustomEmbedding(num_heads=1)
    numeric_embedding_facade = NumericEmbeddingFacade(alibi_custom=alibi_custom)
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
    custom_query_values = torch.tensor([
        [1.0, 4.0]
    ])
    custom_key_values = torch.tensor([
        [2.0, 4.0]
    ])

    output, attn_probs = attention(q, k, v, numeric_embedding_facade=numeric_embedding_facade,
                                   alibi_query_values=custom_query_values,
                                   alibi_key_values=custom_key_values,
                                   padding_and_loss_attention_mask=None)

    expected_attn_probs = torch.tensor([[[
        [0.7101, 0.2899],
        [0.2210, 0.7790]
    ]]])
    expected_output = torch.tensor([[[
        [0.1870, 0.2870, 0.3870],
        [0.3337, 0.4337, 0.5337]
    ]]])
    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)


def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__padding_mask_for_padding_tokens_at_end_work(
        numeric_embedding_facade):
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
    padding_and_loss_attention_mask = torch.tensor([
        [1, 0]
    ])

    attention = StandardAttentionMethod()
    output, attn_probs = attention(
        q, k, v, numeric_embedding_facade=numeric_embedding_facade,
        padding_and_loss_attention_mask=padding_and_loss_attention_mask)

    expected_attn_probs = torch.tensor([[[
        [1.0, 0.0],
        [1.0, 0.0]
    ]]])
    expected_output = torch.tensor([[[
        [1e-1, 2e-1, 3e-1],
        [1e-1, 2e-1, 3e-1],
    ]]])

    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)


def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__padding_mask_for_padding_tokens_at_end_work(
        numeric_embedding_facade):
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
    padding_and_loss_attention_mask = torch.tensor([
        [0, 1]
    ])

    attention = StandardAttentionMethod()
    output, attn_probs = attention(
        q, k, v, numeric_embedding_facade=numeric_embedding_facade,
        padding_and_loss_attention_mask=padding_and_loss_attention_mask)

    expected_attn_probs = torch.tensor([[[
        [0.0, 1.0],
        [0.0, 1.0]
    ]]])
    expected_output = torch.tensor([[[
        [4e-1, 5e-1, 6e-1],
        [4e-1, 5e-1, 6e-1],
    ]]])

    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)

def test__StandardAttentionMethod__dimension3_queryLength2_kvLength2__padding_mask_for_padding_tokens_at_end_work__batchSize2_numHeads2(
        numeric_embedding_facade):
    q = torch.tensor([
        [
            [
                [1e-1, 2e-1, 3e-1],
                [4e-1, 5e-1, 6e-1]
            ],
            [
                [15e-2, 25e-2, 35e-2],
                [45e-2, 55e-2, 65e-2]
            ]
        ],
        [
            [
                [1e-1, 2e-1, 3e-1],
                [4e-1, 5e-1, 6e-1]
            ],
            [
                [15e-2, 25e-2, 35e-2],
                [45e-2, 55e-2, 65e-2]
            ]
        ],
    ])
    k = torch.tensor([
        [
            [
                [1e-1, 2e-1, 3e-1],
                [4e-1, 5e-1, 6e-1]
            ],
            [
                [15e-2, 25e-2, 35e-2],
                [45e-2, 55e-2, 65e-2]
            ]
        ],
        [
            [
                [1e-1, 2e-1, 3e-1],
                [4e-1, 5e-1, 6e-1]
            ],
            [
                [15e-2, 25e-2, 35e-2],
                [45e-2, 55e-2, 65e-2]
            ]
        ],
    ])
    v = torch.tensor([
        [
            [
                [1e-1, 2e-1, 3e-1],
                [4e-1, 5e-1, 6e-1],
            ],
            [
                [15e-2, 25e-2, 35e-2],
                [45e-2, 55e-2, 65e-2],
            ]
        ],
        [
            [
                [1e-1, 2e-1, 3e-1],
                [4e-1, 5e-1, 6e-1],
            ],
            [
                [15e-2, 25e-2, 35e-2],
                [45e-2, 55e-2, 65e-2],
            ]
        ],
    ])
    padding_and_loss_attention_mask = torch.tensor([
        [1, 0],
        [0, 1],
    ])

    attention = StandardAttentionMethod()
    output, attn_probs = attention(
        q, k, v, numeric_embedding_facade=numeric_embedding_facade,
        padding_and_loss_attention_mask=padding_and_loss_attention_mask)

    expected_attn_probs = torch.tensor([
        [
            [
                [1.0, 0.0],
                [1.0, 0.0]
            ],
            [
                [1.0, 0.0],
                [1.0, 0.0]
            ],
        ],
        [
            [
                [0.0, 1.0],
                [0.0, 1.0]
            ],
            [
                [0.0, 1.0],
                [0.0, 1.0]
            ],
        ],
    ])
    expected_output = torch.tensor([
        [
            [
                [1e-1, 2e-1, 3e-1],
                [1e-1, 2e-1, 3e-1],
            ],
            [
                [15e-2, 25e-2, 35e-2],
                [15e-2, 25e-2, 35e-2],
            ],
        ],
        [
            [
                [4e-1, 5e-1, 6e-1],
                [4e-1, 5e-1, 6e-1],
            ],
            [
                [45e-2, 55e-2, 65e-2],
                [45e-2, 55e-2, 65e-2],
            ],
        ],
    ])
    assert torch.allclose(attn_probs, expected_attn_probs, atol=1e-4)
    assert torch.allclose(output, expected_output, atol=1e-4)


