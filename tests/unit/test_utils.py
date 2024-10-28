import torch
from torch import nn
from attention_smithy.utils import create_causal_mask, repeat_module_consecutively

def test__repeat_module_consecutively__repeats_have_different_weights():
    class DummyTestNetwork(nn.Module):
        def __init__(self):
            super(DummyTestNetwork, self).__init__()
            self.fc1 = nn.Linear(5, 10)
            self.fc2 = nn.Linear(10, 5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    dummy_network = DummyTestNetwork()
    layers = repeat_module_consecutively(dummy_network, number_of_repeats=2)
    assert not torch.equal(layers[0].fc1.weight, layers[1].fc1.weight)
    assert not torch.equal(layers[0].fc2.weight, layers[1].fc2.weight)

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