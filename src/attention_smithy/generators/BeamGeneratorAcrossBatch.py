import torch
from torch import nn
from attention_smithy.generators.GeneratorStrategy import GeneratorStrategy

class BeamGeneratorAcrossBatch(GeneratorStrategy):
    """
    A beam search generator strategy class.

    This class performs the same function as the BeamGenerator class, but can work across a full
        batch. The BeamGenerator class assumes a batch size of 1. See the BeamGenerator
        docustring for reference.
    """
    def __init__(self,
                 beam_width:int = 3,
                 length_penalty_alpha:float = 0.6,
                 max_sequence_length:int = 150,
                 ) -> None:
        """
        Args:
            beam_width (int): The number of beams to prioritize on each pass of the token
                generation loop. Default is 3.
            length_penalty_alpha (float): Adjusts the length penalty used in calculating
                each score after a token is added. Default is 0.6.
            max_sequence_length (int): The maximum sequence length to be generated. This
                variable stops beam generation from growing out of hand during early
                training steps.
        """
        super().__init__()
        self.beam_width = beam_width
        self.length_penalty_alpha = length_penalty_alpha
        self.max_sequence_length = max_sequence_length

    def generate_sequence(
        self,
        model: nn.Module,
        end_token: int,
        tgt_input: torch.Tensor,
        maximum_sequence_length: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        See GeneratorContext class.
        """
        self.end_token = end_token
        beams, scores = self._run_input_through_model_and_attach_best_next_tokens_to_separate_beams(model, tgt_input, **kwargs)
        for step in range(tgt_input.size(1), maximum_sequence_length):
            beams, scores = self._add_tokens_to_branching_beams_and_prune_low_scoring_branches(beams, model, scores, step, **kwargs)
            if torch.all(beams[:, -1] == self.end_token) or beams.shape[1] > self.max_sequence_length: break
        best_beams = self._identify_best_beams_from_all_finished_beams(beams, scores)
        return best_beams

    def _add_tokens_to_branching_beams_and_prune_low_scoring_branches(self, beams, model, scores, step, **kwargs):
        top_k_indices_across_beams, top_k_probabilities_across_beams = self._run_beams_through_model_and_get_highest_k_probability_next_tokens(beams, step, model, **kwargs)
        finished_beam_mask = self._identify_all_beams_that_previously_ended_with_end_token(beams)
        expanded_beams = self._add_next_tokens_to_beams(beams, top_k_indices_across_beams, finished_beam_mask)
        expanded_scores = self._update_scores(scores, step, top_k_probabilities_across_beams, finished_beam_mask)
        beams, scores = self._prune_low_scoring_beams_to_only_keep_beam_width_amount(expanded_beams, expanded_scores)
        return beams, scores

    def _identify_all_beams_that_previously_ended_with_end_token(self, beams):
        finished_beam_mask = beams[:, -1] == self.end_token
        finished_beam_mask = finished_beam_mask.unsqueeze(1).repeat_interleave(self.beam_width, dim=0).flatten()
        return finished_beam_mask

    def _run_input_through_model_and_attach_best_next_tokens_to_separate_beams(self, model, tgt_input, **kwargs):
        initial_sequence_length = tgt_input.size(1)
        initial_beam = tgt_input.clone()
        beams, scores = self._initialize_beams_and_corresponding_scores(initial_beam, initial_sequence_length, model, **kwargs)
        return beams, scores

    def _identify_best_beams_from_all_finished_beams(self, beams, scores):
        scores_as_batch = scores.view(-1, self.beam_width)
        batch_size = scores_as_batch.shape[0]
        _, best_beam_indices_by_batch = torch.topk(scores_as_batch, k=1, dim=1)
        best_beam_indices = best_beam_indices_by_batch.flatten() + torch.arange(0, len(scores), self.beam_width).to(beams.device)
        best_beam = beams[best_beam_indices, :]
        return best_beam

    def _prune_low_scoring_beams_to_only_keep_beam_width_amount(self, expanded_beams, expanded_scores):
        expanded_beam_width = self.beam_width*self.beam_width
        batch_size = len(expanded_scores) // expanded_beam_width
        scores_as_batch = expanded_scores.view(batch_size, -1)
        expanded_beam_count = expanded_scores.shape[0]
        _, top_score_indices_by_sample = torch.topk(scores_as_batch, k=self.beam_width, dim=-1)
        batch_index_offset = torch.arange(0, len(expanded_scores), expanded_beam_width).unsqueeze(1).to(expanded_beams.device)
        top_score_indices = (top_score_indices_by_sample + batch_index_offset).flatten()
        top_scores = expanded_scores[top_score_indices]
        top_beams = expanded_beams[top_score_indices, :]
        return top_beams, top_scores

    def _update_scores(self, scores, step, top_k_probabilities_across_beams, finished_beam_mask):
        expanded_old_scores = scores.repeat_interleave(self.beam_width)
        current_sequence_length = step + 1
        length_penalty = ((5 + step) ** self.length_penalty_alpha) / ((5 + 1) ** self.length_penalty_alpha)
        expanded_scores = (expanded_old_scores + top_k_probabilities_across_beams.flatten()) / length_penalty
        expanded_scores[finished_beam_mask] = expanded_old_scores[finished_beam_mask]
        return expanded_scores

    def _add_next_tokens_to_beams(self, beams, top_k_indices_across_beams, finished_beam_mask):
        expanded_beams = beams.repeat_interleave(self.beam_width, dim=0)
        flattened_additions = top_k_indices_across_beams.flatten()
        flattened_additions[finished_beam_mask] = self.end_token
        expanded_beams_with_new_additions = torch.cat((expanded_beams, flattened_additions.unsqueeze(1)), dim=1)
        return expanded_beams_with_new_additions

    def _initialize_beams_and_corresponding_scores(self, initial_beam, initial_sequence_length, model, **kwargs):
        top_k_indices_across_beams, top_k_probabilities_across_beams = self._run_beams_through_model_and_get_highest_k_probability_next_tokens(
            initial_beam, initial_sequence_length - 1, model, **kwargs)
        beams = initial_beam.unsqueeze(1).repeat(1, self.beam_width, 1)
        beams = torch.cat((beams, top_k_indices_across_beams.unsqueeze(2)), dim=2)
        beams = beams.view(-1, beams.size(2))
        scores = top_k_probabilities_across_beams.flatten()
        return beams, scores

    def _run_beams_through_model_and_get_highest_k_probability_next_tokens(self, beams, step,
                                                                         model, **kwargs):
        all_logits = model.forward_decode(beams, **kwargs)
        next_token_logits = all_logits[:, step, :]
        log_probabilities = torch.log(torch.softmax(next_token_logits, dim=-1))
        top_k_probabilities_across_beams, top_k_indices_across_beams = torch.topk(
            log_probabilities, self.beam_width, dim=-1
        )
        return top_k_indices_across_beams, top_k_probabilities_across_beams



