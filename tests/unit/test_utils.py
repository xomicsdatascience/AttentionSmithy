import torch
from attention_smithy.utils import create_causal_mask

def test__causal_mask():
    size = 5
    expected_output = torch.tensor([
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, True, True, False, False],
        [True, True, True, True, False],
        [True, True, True, True, True],
    ])
    output = create_causal_mask(size)
    assert torch.allclose(output, expected_output)