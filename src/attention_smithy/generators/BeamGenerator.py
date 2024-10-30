import torch
from torch import nn
from attention_smithy.generators.GeneratorStrategy import GeneratorStrategy

class BeamGenerator(GeneratorStrategy):
    def __init__(self, beam_width=3, length_penalty_alpha=0.6):
        super().__init__()
        self.beam_width = beam_width
        self.length_penalty_alpha = length_penalty_alpha

    def generate_sequence(
        self,
        model: nn.Module,
        end_token: int,
        tgt_input: torch.Tensor,
        src_embedding: torch.Tensor,
        maximum_sequence_length: int,
        **kwargs,
    ):
        self.end_token = end_token
        beams, scores = self._run_input_through_model_and_attach_best_next_tokens_to_separate_beams(
            model, tgt_input)
        for step in range(tgt_input.size(1), maximum_sequence_length):
            beams, scores = self._add_tokens_to_branching_beams_and_prune_low_scoring_branches(beams, model, scores,
                                                                                               step)
            if torch.all(self.mask_for_beams_that_have_reached_the_end_token):
                best_beam = self._identify_best_beam_from_all_finished_beams(beams, scores)
                return best_beam

        best_beam_index = torch.argmax(scores)
        return beams[best_beam_index]

    def _add_tokens_to_branching_beams_and_prune_low_scoring_branches(self, beams, model, scores, step):
        finished_beams, finished_scores, unfinished_beams, unfinished_scores = self._separate_finished_beams_from_unfinished(
            beams, scores)
        top_k_indices_across_beams, top_k_probabilities_across_beams = self._run_unfinished_beams_through_model_and_get_highest_probability_next_tokens(
            unfinished_beams, step, model)
        expanded_beams = self._add_next_tokens_to_unfinished_beams_then_combine_with_previously_finished_beams(
            finished_beams, finished_scores, top_k_indices_across_beams, unfinished_beams)
        expanded_scores = self._update_unfinished_scores_then_combine_with_finished_scores(finished_scores, step,
                                                                                           top_k_probabilities_across_beams,
                                                                                           unfinished_scores)
        beams, scores = self._prune_low_scoring_beams_to_only_keep_beam_width_amount(expanded_beams,
                                                                                   expanded_scores)
        return beams, scores

    def _run_input_through_model_and_attach_best_next_tokens_to_separate_beams(self, model, tgt_input):
        initial_sequence_length = tgt_input.size(1)
        initial_beam = tgt_input.clone()
        beams, scores = self._initialize_beams_and_corresponding_scores(initial_beam, initial_sequence_length, model)
        self.mask_for_beams_that_have_reached_the_end_token = (beams == self.end_token).any(dim=1)
        return beams, scores

    def _identify_best_beam_from_all_finished_beams(self, beams, scores):
        best_beam_index = torch.argmax(scores)
        best_beam = beams[best_beam_index]
        best_beam = self._remove_trailing_end_tokens_from_best_beam(best_beam)
        return best_beam

    def _remove_trailing_end_tokens_from_best_beam(self, best_beam):
        end_idx = torch.min(torch.where(best_beam == self.end_token)[0]) + 1
        return best_beam[:end_idx]

    def _prune_low_scoring_beams_to_only_keep_beam_width_amount(self, expanded_beams, expanded_scores):
        _, top_score_indices = torch.topk(expanded_scores, k=self.beam_width)
        beams = expanded_beams[top_score_indices].to(torch.int64)
        scores = expanded_scores[top_score_indices]
        self.mask_for_beams_that_have_reached_the_end_token = (beams == self.end_token).any(dim=1)
        return beams, scores

    def _update_unfinished_scores_then_combine_with_finished_scores(self, finished_scores, step,
                                                                    top_k_probabilities_across_beams,
                                                                    unfinished_scores):
        expanded_old_scores = unfinished_scores.repeat_interleave(self.beam_width)
        current_sequence_length = step + 1
        length_penalty = (current_sequence_length + 1) ** self.length_penalty_alpha / (current_sequence_length + 1)
        expanded_new_scores = (expanded_old_scores + top_k_probabilities_across_beams.flatten()) / length_penalty
        expanded_scores = torch.cat((expanded_new_scores, finished_scores), dim=0)
        return expanded_scores

    def _add_next_tokens_to_unfinished_beams_then_combine_with_previously_finished_beams(self, finished_beams,
                                                                               finished_scores,
                                                                               top_k_indices_across_beams,
                                                                               unfinished_beams):
        expanded_unfinished_beams = unfinished_beams.repeat_interleave(self.beam_width, dim=0)
        expanded_beams = torch.cat((expanded_unfinished_beams, finished_beams), dim=0)
        beam_additions = torch.cat(
            (top_k_indices_across_beams.flatten(), torch.tensor([self.end_token for i in range(len(finished_scores))])))
        expanded_beams_with_new_additions = torch.cat((expanded_beams, beam_additions.unsqueeze(1)), dim=1)
        return expanded_beams_with_new_additions

    def _separate_finished_beams_from_unfinished(self, beams, scores):
        finished_beams = beams[self.mask_for_beams_that_have_reached_the_end_token]
        finished_scores = scores[self.mask_for_beams_that_have_reached_the_end_token]
        unfinished_beams = beams[~self.mask_for_beams_that_have_reached_the_end_token]
        unfinished_scores = scores[~self.mask_for_beams_that_have_reached_the_end_token]
        return finished_beams, finished_scores, unfinished_beams, unfinished_scores

    def _initialize_beams_and_corresponding_scores(self, initial_beam, initial_sequence_length, model):
        top_k_indices_across_beams, top_k_probabilities_across_beams = self._run_unfinished_beams_through_model_and_get_highest_probability_next_tokens(
            initial_beam, initial_sequence_length - 1, model)
        beams = initial_beam.repeat(self.beam_width, 1)
        beams = torch.cat((beams, top_k_indices_across_beams.transpose(0, 1)), dim=1)
        scores = top_k_probabilities_across_beams.flatten()
        return beams, scores

    def _run_unfinished_beams_through_model_and_get_highest_probability_next_tokens(self, beams, step,
                                                                         model):
        all_logits = model.forward_decode(beams)
        next_token_logits = all_logits[:, step, :]
        log_probabilities = torch.log(torch.softmax(next_token_logits, dim=-1))
        top_k_probabilities_across_beams, top_k_indices_across_beams = torch.topk(
            log_probabilities, self.beam_width, dim=-1
        )
        return top_k_indices_across_beams, top_k_probabilities_across_beams



