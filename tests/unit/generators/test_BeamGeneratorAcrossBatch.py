import torch
from torch import nn
import pytest
import random
from attention_smithy.generators import GeneratorContext, GeneratorModuleAbstractClass
from attention_smithy.test_functions import generate_using_pretrained_model

@pytest.fixture
def beam_generator_context():
    return GeneratorContext(method="beam")

@pytest.fixture
def vocab_size():
    return 60

@pytest.fixture
def batch_size():
    return 12

@pytest.fixture
def expected_output(start_token, end_token, vocab_size, batch_size):
    random.seed(0)
    expected_list = list(range(6, vocab_size-10, 3))
    single_sample = torch.tensor([[start_token] + expected_list + [end_token, end_token]])
    return single_sample.repeat(batch_size, 1)

@pytest.fixture
def dummy_model(vocab_size, expected_output, end_token):
    class DummyBeamSearchModel(GeneratorModuleAbstractClass):
        def __init__(self, vocab_size, desired_sequence):
            super(DummyBeamSearchModel, self).__init__()
            self.vocab_size = vocab_size
            self.token_embedding = nn.Embedding(vocab_size, vocab_size)
            self.token_embedding.weight.data = self._create_dummy_weights(desired_sequence)

        def _create_dummy_weights(self, desired_sequence):
            weights = self._initialize_dummy_weights()
            high_score = 10.0
            mid_score = 8.0
            low_score = 6.0
            meh_score = 2.0
            self._ensure_optimal_path_is_not_found_through_the_first_greedy_option(desired_sequence, high_score,
                                                                                  low_score, mid_score, weights)
            for i, token in enumerate(desired_sequence[1:-1]):
                next_token = token + 3
                if token == desired_sequence[-2]:
                    next_token = end_token
                weights[token, next_token] = high_score
                weights[token, next_token + 1] = mid_score
                weights[token, next_token + 2] = low_score
                self._set_side_branch_paths_to_temporary_slight_increase_in_weights(meh_score, token, weights)
            return weights

        def _set_side_branch_paths_to_temporary_slight_increase_in_weights(self, meh_score, token, weights):
            weights[token + 1, token + 3] = meh_score
            weights[token + 1, token + 4] = meh_score
            weights[token + 2, token + 3] = meh_score
            weights[token + 2, token + 4] = meh_score

        def _initialize_dummy_weights(self):
            weights = torch.zeros((vocab_size, vocab_size))
            self._set_weights_at_end_of_program_to_point_to_end_token(weights)
            return weights

        def _set_weights_at_end_of_program_to_point_to_end_token(self, weights):
            weights[-10:, end_token] = 1.0
            weights[end_token + 1, end_token] = 1.0
            weights[end_token + 1, end_token + 2] = 1.0
            weights[end_token + 2, end_token] = 1.0

        def _ensure_optimal_path_is_not_found_through_the_first_greedy_option(self, desired_sequence, high_score,
                                                                              low_score, mid_score, weights):
            first_token = desired_sequence[0]
            second_token = desired_sequence[1]
            weights[first_token, second_token] = mid_score
            weights[first_token, 2] = low_score
            weights[first_token, second_token + 2] = high_score

        def forward_decode(self, input_tokens):
            embedded_tokens = self.token_embedding(input_tokens)
            return embedded_tokens

    return DummyBeamSearchModel(vocab_size, expected_output[0][:-1])

@pytest.fixture
def start_token():
    return 1

@pytest.fixture
def end_token():
    return 2


@pytest.fixture
def tgt_input_tensor(start_token, batch_size):
    return torch.full((batch_size, 1), start_token)

def test__BeamGeneratorAcrossBatch(beam_generator_context, dummy_model, tgt_input_tensor, end_token, expected_output):
    args = (dummy_model, end_token, tgt_input_tensor)
    with torch.no_grad():
        output = beam_generator_context.generate_sequence(*args)
        assert torch.allclose(output, expected_output)

def test__BeamGeneratorAcrossBatch__using_pretrained_model():
    generate_using_pretrained_model(method='beam')
