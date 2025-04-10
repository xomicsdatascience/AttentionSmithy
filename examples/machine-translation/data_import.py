import torch
from torch.utils.data import Dataset, BatchSampler, Sampler
import torch.distributed as dist
import mmap
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import random

data_directory = 'data'

class MachineTranslationDataModule(pl.LightningDataModule):
    def __init__(self,
                 en_filepath_suffix: str,
                 de_filepath_suffix: str,
                 maximum_length: int,
                 batch_size: int,
                 num_training_samples: int = None,
                 ):

        super().__init__()
        self.en_filepath_suffix = en_filepath_suffix
        self.de_filepath_suffix = de_filepath_suffix
        self.maximum_length = maximum_length
        self.batch_size = batch_size
        self.num_training_samples = num_training_samples
        self.de_pad_token, self.en_pad_token, self.de_vocab_size, self.en_vocab_size = self.get_tokenizer_values()

    def setup(self, stage=None):
        self.train_dataset = LineIndexDataset(f'{data_directory}/train{self.de_filepath_suffix}', f'{data_directory}/train{self.en_filepath_suffix}', self.num_training_samples)
        self.val_dataset = LineIndexDataset(f'{data_directory}/val{self.de_filepath_suffix}', f'{data_directory}/val{self.en_filepath_suffix}', self.num_training_samples)
        self.test_dataset = LineIndexDataset(f'{data_directory}/test{self.de_filepath_suffix}', f'{data_directory}/test{self.en_filepath_suffix}', self.num_training_samples)

    def train_dataloader(self):
        sampler = torch.utils.data.RandomSampler(self.train_dataset)
        batch_sampler = LengthBatchSampler(sampler, batch_size=self.batch_size, drop_last=False,
                                           dataset=self.train_dataset)
        return DataLoader(self.train_dataset, batch_sampler=batch_sampler, collate_fn=self._collate_function)

    def val_dataloader(self):
        sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        batch_sampler = LengthBatchSampler(sampler, batch_size=self.batch_size, drop_last=False,
                                           dataset=self.val_dataset)
        return DataLoader(self.val_dataset, batch_sampler=batch_sampler, collate_fn=self._collate_function)

    def test_dataloader(self):
        sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        batch_sampler = LengthBatchSampler(sampler, batch_size=self.batch_size, drop_last=False,
                                           dataset=self.test_dataset)
        return DataLoader(self.test_dataset, batch_sampler=batch_sampler, collate_fn=self._collate_function)

    def _collate_function(self, batch):
        input_tensors, expected_output_tensors = zip(*batch)
        src_input_tensor = nn.utils.rnn.pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.de_pad_token,
        )
        output_tensor = nn.utils.rnn.pad_sequence(
            expected_output_tensors,
            batch_first=True,
            padding_value=self.en_pad_token,
        )
        src_input_tensor = src_input_tensor[:, :self.maximum_length]
        output_tensor = output_tensor[:, :self.maximum_length]
        tgt_input_tensor = output_tensor[:, :-1]
        expected_output_tensor = output_tensor[:, 1:]

        src_padding_mask = (src_input_tensor != self.de_pad_token)
        tgt_padding_mask = (tgt_input_tensor != self.en_pad_token)

        return src_input_tensor, tgt_input_tensor, expected_output_tensor, src_padding_mask, tgt_padding_mask

    def get_tokenizer_values(self):
        de_tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        return de_tokenizer.convert_tokens_to_ids(de_tokenizer.pad_token), \
            en_tokenizer.convert_tokens_to_ids(en_tokenizer.pad_token), \
            de_tokenizer.vocab_size, en_tokenizer.vocab_size

class LineIndexDataset(Dataset):
    def __init__(self, input_filepath, expected_output_filepath, num_training_samples):
        self.input_file = MappedFile(input_filepath)
        self.expected_output_file = MappedFile(expected_output_filepath)
        self.num_training_samples = num_training_samples
        lengths = []
        with open(input_filepath, 'r') as input_file, open(expected_output_filepath, 'r') as expected_output_file:
            for input_line, expected_output_line in zip(input_file, expected_output_file):
                input_token_count = input_line.count(',')
                output_token_count = expected_output_line.count(',')
                lengths.append(input_token_count + output_token_count)
        self.lengths = lengths

    def __len__(self):
        if self.num_training_samples == None:
            return len(self.lengths)
        else:
            return min(len(self.lengths), self.num_training_samples)

    def __getitem__(self, idx):
        input_tensor = [int(token) for token in self.input_file.get_line(idx).strip().split(',')]
        expected_output_tensor = [int(token) for token in self.expected_output_file.get_line(idx).strip().split(',')]
        return torch.tensor(input_tensor), torch.tensor(expected_output_tensor)

class MappedFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.mm = None
        self.line_offsets = []
        self.load()

    def load(self):
        with open(self.file_path, 'r') as file:
            self.mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            offset = 0
            while True:
                newline_pos = self.mm.find(b'\n', offset)
                if newline_pos == -1:
                    break
                self.line_offsets.append(offset)
                offset = newline_pos + 1
            self.line_offsets.append(offset)

    def get_line(self, line_number):
        if line_number < 0 or line_number > len(self.line_offsets):
            raise ValueError("Invalid line number")
        start = self.line_offsets[line_number]
        end = self.line_offsets[line_number + 1]
        return self.mm[start:end].decode('utf-8').strip()


class LengthBatchSampler(BatchSampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False, dataset=None):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                f"Expected sampler to be an instance of torch.utils.data.Sampler, got {type(sampler)} instead.")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        if hasattr(sampler, 'data_source') and hasattr(sampler.data_source, 'lengths'):
            self.lengths = sampler.data_source.lengths
        elif dataset is not None and hasattr(dataset, 'lengths'):
            self.lengths = dataset.lengths
        else:
            raise ValueError("Sampler must have a dataset with 'lengths' attribute or a dataset must be provided.")

    def __iter__(self):
        indices = list(self.sampler)
        indices.sort(key=lambda i: self.lengths[i])

        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches.pop()

        # Shuffle batches if using a random sampler
        if isinstance(self.sampler, torch.utils.data.RandomSampler):
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        return len(self.sampler) // self.batch_size if self.drop_last else (len(self.sampler) + self.batch_size - 1) // self.batch_size

class LabelSmoothingLoss(nn.Module):
    """
    A class that performs loss functionality for the machine translation project. While a standard
        KLDivLoss could be directly applied, the writers of the Attention Is All You Need paper outlined
        a label smoothing pre-process step that is also performed here.

    In training, we want the model to accurately predict the next token in a sequence. At a high level,
        you could say that the correct next token is 100% right, and all other tokens are 0% right. The
        model will train just fine if you calculate the loss under this assumption. However, the authors
        determined to make the correct next token "probably" right, rather than "absolutely" right. In
        short, the correct next token is 90% (adjustable) right, and the remaining 10% probability is
        evenly distributed across the remaining tokens. This "smooths" the label probabilities, hence
        the name.

    This process hurts the training loss score, but improves the final BLEU score, the primary metric used
        in evaluating translation models from a human's perspective.
    """


    def __init__(self, padding_token_idx, confidence_probability_score):
        """
        Args:
            padding_token_idx (int): The padding token ID. The probability for this specific token
                should always be 0 - it should never be considered in predicting the next token.
            confidence_probability_score (float): The confidence probability assigned to the correct
                token (in place of 1.0, or 100%).
        """

        super().__init__()
        self.padding_token_idx = padding_token_idx
        self.confidence_probability_score = confidence_probability_score
        self.inverse_probability_score = 1.0 - confidence_probability_score
        self.negating_probability_score = 0.0
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, vocab_logits, expected_output_tokens, batch_idx):
        """
        Note: In training, the tgt input and the expected output are effectively shifted windows of
            each other. So if a single token sentence is represented by `sentence`, tgt input is
            `sentence[:-1]` and the expected output is `sentence[1:]`.

        Args:
            vocab_logits (torch.Tensor): The output of the machine translation model. This should
                be of shape (batch_size, (sequence_length-1), vocab_size). It represents the
                probability calculated by the model for each possible next token of the sequence.
            expected_output_tokens (torch.Tensor): A tensor representing the next tokens to be
                predicted, of shape (batch_size, (sequence_length-1)).
        """
        self.device = vocab_logits.device
        smooth_label_expected_distribution = self._create_smooth_label_expected_distribution(expected_output_tokens, *vocab_logits.shape)
        vocab_logits_reshaped, smooth_label_expected_distribution_reshaped = self._reshape_to_remove_padding_token_targets(
            vocab_logits, smooth_label_expected_distribution, expected_output_tokens)
        return self.criterion(vocab_logits_reshaped, smooth_label_expected_distribution_reshaped)

    def _create_smooth_label_expected_distribution(self, expected_output_tokens, batch_size, tgt_sequence_length, tgt_vocab_size):
        smooth_label_expected_distribution = self._initialize_label_distribution_with_low_confidence_values(batch_size, tgt_sequence_length, tgt_vocab_size)
        self._set_target_token_to_high_confidence_value(expected_output_tokens, smooth_label_expected_distribution)
        self._negate_confidence_values_for_padding_tokens(expected_output_tokens, smooth_label_expected_distribution)
        return smooth_label_expected_distribution

    def _initialize_label_distribution_with_low_confidence_values(self, batch_size, tgt_sequence_length, tgt_vocab_size):
        number_of_non_target_non_padding_tokens = tgt_vocab_size - 2
        dispersed_inverse_probability_score = self.inverse_probability_score / number_of_non_target_non_padding_tokens
        smooth_label_expected_distribution = torch.full((batch_size, tgt_sequence_length, tgt_vocab_size),
                                                        dispersed_inverse_probability_score, device=self.device)
        return smooth_label_expected_distribution

    def _set_target_token_to_high_confidence_value(self, expected_output_tokens, smooth_label_expected_distribution):
        smooth_label_expected_distribution.scatter_(-1, expected_output_tokens.unsqueeze(-1), self.confidence_probability_score)

    def _negate_confidence_values_for_padding_tokens(self, expected_output_tokens, smooth_label_expected_distribution):
        smooth_label_expected_distribution[:, :, self.padding_token_idx] = self.negating_probability_score

    def _reshape_to_remove_padding_token_targets(self,
                                                    vocab_logits,
                                                    smooth_label_expected_distribution,
                                                    expected_output_tokens,
                                                ):
        batch_size, tgt_sequence_length, tgt_vocab_size = vocab_logits.shape
        vocab_logits_reshaped = vocab_logits.reshape(batch_size*tgt_sequence_length, tgt_vocab_size)
        smooth_label_expected_distribution_reshaped = smooth_label_expected_distribution.reshape(batch_size*tgt_sequence_length, tgt_vocab_size)
        padding_token_mask = expected_output_tokens.flatten() == self.padding_token_idx
        vocab_logits_reshaped = vocab_logits_reshaped[~padding_token_mask]
        smooth_label_expected_distribution_reshaped = smooth_label_expected_distribution_reshaped[~padding_token_mask]
        return vocab_logits_reshaped, smooth_label_expected_distribution_reshaped