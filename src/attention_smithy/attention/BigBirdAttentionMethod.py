import torch
import torch.nn as nn
from torch.nn import functional as F
from attention_smithy.utils import create_causal_mask


class BigBirdAttentionMethod(nn.Module):
    """
    A PyTorch module implementing the Big Bird attention mechanism.

    Big Bird attention was designed to circumvent the primary bottleneck of attention, namely,
        the constraints on memory that quadratically worsen with longer context windows. This stems from
        the attention matrix, where every token from the query "attends to" every token from the key (n*m).

    Big Bird utilizes a sparse attention matrix format. It utilizes three categories of attention, namely
        global, local, and random.

     - Global attention: These are tokens that should attend/be attended to by all tokens. In
        practice, the standard attention method uses global attention for all tokens. In BigBird, only key
        tokens are established with this global distinction, and are generally the most memory intensive.
     - Local attention: Only applicable for self-attention. Tokens attend/are attended to other tokens
        their vicinity.
     - Random attention: Tokens that are chosen to attend to each other at random. This is one of the prime
        distinctions from the Longformer method, and provides a kind of random dispersal of attention that
        facilitates long-term broad connectivity between attention blocks.

    One element of Big Bird attentions is the "blockification" of the attention matrix, such that entire
        groups of neighboring tokens are combined into "blocks." This eases the process of selecting
        specific tokens prior to the multiplication step, particularly for random attention.

    This package selects attention blocks via direct indexing, which deviates from the method described
        in the  original Big Bird paper.

    DEV NOTE: For developers looking to understand the code, it is strongly advised that they look at the
        attention blocks outlined in the test file under
        `AttentionSmithy/tests/unit/attention/test_BigBirdAttentionMethod.py`. As a visual aid, you can
        search for "1" values, which will highlight the attention blocks being attended to in each test.

    DEV NOTE: A distinction is made between "global" and "sparse" query blocks. That is because query blocks
        that contain global tokens effectively perform standard attention, and only their "sparse" counterparts
        require calculations for global, local and random attention. Thus, their calculations are different.

    """

    def __init__(
        self,
        block_size_query: int,
        block_size_kv: int,
        local_window_extension_length: int,
        num_random_blocks: int = 0,
        is_causal_masking: bool = False,
        max_block_limit: int = 0,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the Big Bird attention module.

        Args:
            block_size_query (int): The number of tokens that should be in each "block" of the query. For example,
                if there are 20 tokens coming in from the query matrix, and the block size is 4, the query will
                be broken into 5 blocks.
            block_size_kv (int): See `block_size_query`, but for the key matrix instead.
            local_window_extension_length (int): The number of blocks attended to locally. A value of 0 will disable
                local attention; 1 will mean each block attends to itself; 2 will mean each block attends to itself
                and then its two neighbors; and so on.
            num_random_blocks (int, optional): The number of random (key) blocks. Defaults to 0.
            is_causal_masking (bool, optional): Whether to apply causal masking, where each token only attends
                to previous tokens. This class has a method for generating a causal mask that accounts for
                blocking across global, local, and random attention.
            max_block_limit (int, optional): The maximum block limit. Priority given to global blocks first, then
                local, then random. The purpose is to impose a constraint on the sparse matrix size. Defaults to
                0 (no limit).
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """

        super().__init__()
        self.block_size_query = block_size_query
        self.block_size_kv = block_size_kv
        self.local_window_extension_length = local_window_extension_length
        self.num_random_blocks = num_random_blocks
        self.is_causal_masking = is_causal_masking
        self.max_block_limit = max_block_limit
        self.global_dropout = nn.Dropout(dropout)
        self.sparse_dropout = nn.Dropout(dropout)

        self.global_softmax_permutation_shape = (0, 1, 2, 4, 3, 5)
        self.global_softmax_reshape_shape = (0, 1, 2, 4, 3, 5)
        self.sparse_softmax_permutation_shape = (0, 1, 3, 4, 2, 5)
        self.sparse_softmax_reshape_shape = (0, 1, 4, 2, 3, 5)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        numeric_embedding_facade,
        global_tokens_query: torch.Tensor,
        global_tokens_kv: torch.Tensor,
        padding_and_loss_attention_mask: torch.Tensor,
        kv_index_table=None,
    ) -> torch.Tensor:
        """
        Forward pass of the Big Bird attention module.

        Args:
            q (torch.Tensor): The query tensor embedding, of shape
                (batch_size, number_of_heads, query_sequence_length, head_dimension)
            k (torch.Tensor): The key tensor embedding, of shape
                (batch_size, number_of_heads, kv_sequence_length, head_dimension)
            v (torch.Tensor): The value tensor embedding, of shape
                (batch_size, number_of_heads, kv_sequence_length, head_dimension)
            numeric_embedding_facade (NumericEmbeddingFacade): Class that facilitates positional embeddings.
                TODO: Not currently used in Big Bird Attention. Would apply for ALiBi and Relative positional
                embeddings.
            global_tokens_query (torch.Tensor): A mask for the query tensor marking global tokens, of shape
                (batch_size, query_sequence_length)
            global_tokens_kv (torch.Tensor): A mask for the key and value tensors marking global tokens, of shape
                (batch_size, kv_sequence_length)
            padding_and_loss_attention_mask (torch.Tensor, optional): The padding attention mask, of shape
                (batch_size, kv_sequence_length).
            kv_index_table (torch.Tensor): Used for testing purposes only. Represents the direct index of blocks to
                be used for attention. Each row length corresponds to the number of sparse query blocks. The number
                of rows corresponds to the number of kv blocks. Duplicate blocks are calculated, but then only one
                is kept in the end. This is generated by the module, so this parameter is only used for specific
                tests.

        Returns:
            torch.Tensor: The output tensor, of shape (batch_size, num_heads, query_length, head_dim)
        """
        self._initialize_parameters(q, k, global_tokens_query, global_tokens_kv)
        key_blocks, query_blocks, value_blocks = (
            self._reshape_all_tensors_for_block_and_batch_calculations(k, q, v)
        )
        global_query_blocks, sparse_query_blocks, final_outputs_flat = (
            self._split_queries_into_global_and_sparse_blocks(query_blocks)
        )
        global_attention_softmax_weights, global_output_flat = (
            self._calculate_global_query_attention(
                global_query_blocks,
                key_blocks,
                padding_and_loss_attention_mask,
                query_blocks,
                value_blocks,
            )
        )
        final_outputs_flat[:, self.global_query_block_indices_flat] = global_output_flat
        if self.num_global_query_blocks == self.num_blocks_query:
            return (
                self._reshape_final_outputs(final_outputs_flat, q),
                global_attention_softmax_weights,
            )

        sparse_attention_softmax_weights, sparse_output_flat = (
            self._calculate_sparse_query_attention(
                key_blocks,
                kv_index_table,
                padding_and_loss_attention_mask,
                sparse_query_blocks,
                value_blocks,
            )
        )

        final_outputs_flat[:, self.sparse_query_block_indices_flat] = sparse_output_flat
        return self._reshape_final_outputs(final_outputs_flat, q), (
            sparse_attention_softmax_weights,
            global_attention_softmax_weights,
        )

    def _calculate_global_query_attention(
        self,
        global_query_blocks,
        key_blocks,
        padding_and_loss_attention_mask,
        query_blocks,
        value_blocks,
    ):
        global_attention_scores = self._calculate_global_attention_scores(
            global_query_blocks, key_blocks, padding_and_loss_attention_mask
        )
        global_attention_softmax_weights = self._calculate_attention_softmax_weights(
            global_attention_scores,
            self.global_softmax_permutation_shape,
            self.global_softmax_reshape_shape,
            self.kv_length,
        )
        global_attention_softmax_weights = self.sparse_dropout(
            global_attention_softmax_weights
        )
        global_output_flat = self._matmul_global_blocks_by_value_blocks(
            global_attention_softmax_weights, query_blocks, value_blocks
        )
        return global_attention_softmax_weights, global_output_flat

    def _calculate_sparse_query_attention(
        self,
        key_blocks,
        kv_index_table,
        padding_and_loss_attention_mask,
        sparse_query_blocks,
        value_blocks,
    ):
        kv_index_table, sparse_attention_scores = (
            self._calculate_sparse_attention_scores(
                kv_index_table, sparse_query_blocks, key_blocks, padding_and_loss_attention_mask
            )
        )
        sparse_attention_softmax_weights = self._calculate_attention_softmax_weights(
            sparse_attention_scores,
            self.sparse_softmax_permutation_shape,
            self.sparse_softmax_reshape_shape,
            self.num_key_blocks_per_sparse_query_block * self.block_size_kv,
        )
        sparse_attention_softmax_weights = self.sparse_dropout(
            sparse_attention_softmax_weights
        )
        sparse_output_flat = self._matmul_sparse_blocks_by_value_blocks(
            kv_index_table, sparse_attention_softmax_weights, value_blocks
        )
        return sparse_attention_softmax_weights, sparse_output_flat

    def _initialize_parameters(self, q, k, global_tokens_query, global_tokens_kv):
        self.batch_size = q.size(0)
        self.num_heads = q.size(1)
        self.query_length = q.size(2)
        self.kv_length = k.size(2)
        self.head_dim = q.size(3)
        self.num_blocks_query = self.query_length // self.block_size_query
        self.num_blocks_kv = self.kv_length // self.block_size_kv
        self._raise_variable_errors_if_present()

        self.global_query_block_indices = self.set_global_blocks(
            global_tokens_query, self.block_size_query
        )
        if self.max_block_limit != 0:
            self.global_query_block_indices = self.global_query_block_indices[
                :, : self.max_block_limit
            ]
        self.global_kv_block_indices = self.set_global_blocks(
            global_tokens_kv, self.block_size_kv
        )
        self.pad_attention_square = torch.full(
            (self.block_size_query, self.block_size_kv), float("-inf")
        ).to(global_tokens_query.device)
        self.causal_attention_square_mask = create_causal_mask(
            self.block_size_query
        ).to(global_tokens_query.device)

        all_blocks = (
            torch.arange(self.num_blocks_query)
            .repeat(self.batch_size, 1)
            .to(global_tokens_query.device)
        )
        self.num_global_query_blocks = self.global_query_block_indices.size(1)
        if not self.num_global_query_blocks:
            self.sparse_query_block_indices = all_blocks.reshape(self.batch_size, -1)
        else:
            global_blocks_mask = self.global_query_block_indices.unsqueeze(
                2
            ) == all_blocks.unsqueeze(1)
            global_blocks_mask = global_blocks_mask.any(dim=1)
            max_false_values = (~global_blocks_mask).sum(dim=1).max().item()
            if max_false_values == 0:
                self.sparse_query_block_indices = torch.zeros(
                    size=(self.batch_size, 0), dtype=torch.int64
                ).to(global_tokens_query.device)
            else:
                sparse_query_block_indices = torch.full(
                    (self.batch_size, max_false_values), 0, dtype=all_blocks.dtype
                ).to(global_tokens_query.device)
                for i in range(self.batch_size):
                    num_false_values = (~global_blocks_mask[i]).sum().item()
                    sparse_query_block_indices[i, :num_false_values] = all_blocks[
                        i, ~global_blocks_mask[i]
                    ]
                    sparse_query_block_indices[i, num_false_values:] = (
                        sparse_query_block_indices[i, num_false_values - 1]
                    )
                self.sparse_query_block_indices = sparse_query_block_indices
        increments = (
            torch.arange(self.batch_size).to(global_tokens_query.device)
            * self.num_blocks_query
        )
        self.global_query_block_indices_flat = (
            self.global_query_block_indices + increments[:, None]
        ).flatten()
        self.sparse_query_block_indices_flat = (
            self.sparse_query_block_indices + increments[:, None]
        ).flatten()
        self.num_sparse_query_blocks = self.sparse_query_block_indices.size(1)

    def _raise_variable_errors_if_present(self):
        if self.is_causal_masking and self.block_size_query != self.block_size_kv:
            raise RuntimeError(
                f"query block size ({self.block_size_query}) must match kv block size ({self.block_size_kv}) when using causal masking to enable masking on the diagonal."
            )
        if self.block_size_query * self.num_blocks_query != self.query_length:
            raise RuntimeError(
                f"query input sequence length must be divisible by the size of query blocks ({self.block_size_query}). Query input sequence length: {self.query_length}"
            )
        if self.block_size_kv * self.num_blocks_kv != self.kv_length:
            raise RuntimeError(
                f"key and value input sequence length must be divisible by the size of key/value blocks ({self.block_size_kv}). Key/Value input sequence length: {self.kv_length}"
            )

    def set_global_blocks(self, global_tokens, block_size):
        global_blocks = []
        for tensor in global_tokens:
            indices = (
                torch.nonzero(tensor.reshape(-1, block_size).sum(dim=1))
                .squeeze()
                .to(global_tokens.device)
            )
            if indices.dim() == 0:
                indices = torch.tensor([indices])
            global_blocks.append(indices)
        global_blocks = _pad_sequence_with_duplicates(global_blocks)
        return global_blocks.to(global_tokens.device)

    def _matmul_sparse_blocks_by_value_blocks(
        self, kv_index_table, sparse_attention_softmax_weights, value_blocks
    ):
        value_blocks_flat = value_blocks.reshape(
            self.num_heads, -1, self.block_size_kv, self.head_dim
        )
        kv_index_table_flat = kv_index_table.view(-1, kv_index_table.size(-1))
        value_blocks_by_index_flat = value_blocks_flat[:, kv_index_table_flat]
        value_blocks_by_index = value_blocks_by_index_flat.view(
            self.num_heads,
            self.batch_size,
            self.num_key_blocks_per_sparse_query_block,
            kv_index_table.size(2),
            self.block_size_kv,
            self.head_dim,
        )
        sparse_block_output = torch.matmul(
            sparse_attention_softmax_weights, value_blocks_by_index
        ).transpose(2, 3)
        sparse_output = torch.sum(sparse_block_output, dim=3)
        sparse_output_flat = sparse_output.reshape(
            self.num_heads, -1, self.block_size_query, self.head_dim
        )
        return sparse_output_flat

    def _calculate_sparse_attention_scores(
        self, kv_index_table, sparse_query_blocks, key_blocks, padding_and_loss_attention_mask
    ):
        if kv_index_table == None:
            kv_index_table = self._create_kv_index_table_for_local_global_random_block_selection().to(
                key_blocks.device
            )
        if self.max_block_limit != 0:
            kv_index_table = kv_index_table[:, : self.max_block_limit]
        self.num_key_blocks_per_sparse_query_block = kv_index_table.size(1)
        kv_index_table, first_occurrence_mask = _determine_first_occurrences(
            kv_index_table
        )
        sparse_attention_scores = (
            self._calculate_all_sparse_attention_scores_from_index(
                sparse_query_blocks, key_blocks, kv_index_table, padding_and_loss_attention_mask
            )
        )
        sparse_attention_scores = self._pad_duplicate_key_blocks(
            first_occurrence_mask, sparse_attention_scores
        )
        return kv_index_table, sparse_attention_scores

    # SOMEDAY/MAYBE: replace duplicate blocks with random non-duplicate? (presently treating duplicates as padded blocks)
    def _pad_duplicate_key_blocks(self, first_occurrence_mask, sparse_attention_scores):
        sparse_attention_scores_flat = sparse_attention_scores.reshape(
            self.num_heads,
            -1,
            self.num_sparse_query_blocks,
            self.block_size_query,
            self.block_size_kv,
        )
        first_occurrence_mask_flat = first_occurrence_mask.view(
            -1, first_occurrence_mask.size(2)
        )
        sparse_attention_scores_flat[:, ~first_occurrence_mask_flat] = (
            self.pad_attention_square
        )
        sparse_attention_scores = sparse_attention_scores_flat.reshape(
            sparse_attention_scores.shape
        )
        return sparse_attention_scores

    def _reshape_all_tensors_for_block_and_batch_calculations(self, k, q, v):
        key_blocks, query_blocks, value_blocks = self._reshape_tensors_into_blocks(
            k, q, v
        )
        key_blocks, query_blocks, value_blocks = (
            self._transpose_head_num_and_batch_size(
                key_blocks, query_blocks, value_blocks
            )
        )
        return key_blocks, query_blocks, value_blocks

    def _transpose_head_num_and_batch_size(
        self, key_blocks, query_blocks, value_blocks
    ):
        query_blocks = query_blocks.transpose(0, 1)
        key_blocks = key_blocks.transpose(0, 1)
        value_blocks = value_blocks.transpose(0, 1)
        return key_blocks, query_blocks, value_blocks

    def _reshape_tensors_into_blocks(self, k, q, v):
        query_blocks = self._blockify_tensor(
            q, self.num_blocks_query, self.block_size_query
        )
        key_blocks = self._blockify_tensor(
            k, self.num_blocks_kv, self.block_size_kv
        ).transpose(-2, -1)
        value_blocks = self._blockify_tensor(v, self.num_blocks_kv, self.block_size_kv)
        return key_blocks, query_blocks, value_blocks

    def _reshape_final_outputs(self, final_outputs_flat, q):
        final_outputs = final_outputs_flat.view(
            self.num_heads, self.batch_size, -1, self.block_size_query, self.head_dim
        )
        final_outputs = final_outputs.transpose(0, 1)
        final_outputs = final_outputs.view(q.shape)
        return final_outputs

    def _matmul_global_blocks_by_value_blocks(
        self, global_attention_softmax_weights, query_blocks, value_blocks
    ):

        global_output = torch.matmul(
            global_attention_softmax_weights, value_blocks.unsqueeze(2)
        ).sum(dim=3)
        global_output_flat = global_output.reshape(
            self.num_heads, -1, self.block_size_query, self.head_dim
        )
        return global_output_flat

    def _split_queries_into_global_and_sparse_blocks(self, query_blocks):
        query_blocks_flat = query_blocks.reshape(
            self.num_heads,
            self.batch_size * self.num_blocks_query,
            self.block_size_query,
            self.head_dim,
        )
        global_query_blocks_flat = query_blocks_flat[
            :, self.global_query_block_indices_flat
        ]
        sparse_query_blocks_flat = query_blocks_flat[
            :, self.sparse_query_block_indices_flat
        ]

        global_query_blocks = global_query_blocks_flat.view(
            self.num_heads, self.batch_size, -1, self.block_size_query, self.head_dim
        )
        sparse_query_blocks = sparse_query_blocks_flat.view(
            self.num_heads, self.batch_size, -1, self.block_size_query, self.head_dim
        )
        final_outputs_flat = torch.empty_like(query_blocks_flat)
        return global_query_blocks, sparse_query_blocks, final_outputs_flat

    def _blockify_tensor(self, tensor, num_blocks, block_size):
        return tensor.reshape(
            self.batch_size, self.num_heads, num_blocks, block_size, self.head_dim
        )

    def _calculate_global_attention_scores(
        self, global_query_blocks, key_blocks, padding_and_loss_attention_mask
    ):
        attention_scores = torch.matmul(
            global_query_blocks.unsqueeze(3), key_blocks.unsqueeze(2)
        ) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_scores_flat = attention_scores.reshape(
            self.num_heads,
            self.batch_size * self.num_global_query_blocks,
            self.num_blocks_kv,
            self.block_size_query,
            self.block_size_kv,
        )
        if self.is_causal_masking:
            mask_for_diagonal_blocks, mask_for_future_blocks = (
                self._find_future_and_diagonal_masks_for_global_calculations()
            )
            attention_scores = self._apply_causal_masking(
                attention_scores,
                mask_for_diagonal_blocks,
                mask_for_future_blocks,
                self.num_global_query_blocks,
                self.num_blocks_kv,
            )
        if padding_and_loss_attention_mask != None:
            padding_and_loss_attention_mask_blocks = self.split_padding_and_loss_attention_mask_by_blocks(
                padding_and_loss_attention_mask
            )
            attention_scores = (
                self.apply_padding_and_loss_attention_mask_to_global_attention_scores(
                    attention_scores, padding_and_loss_attention_mask_blocks
                )
            )
        return attention_scores

    def apply_padding_and_loss_attention_mask_to_global_attention_scores(
        self, attention_scores, padding_and_loss_attention_mask_blocks
    ):
        attention_scores = attention_scores.transpose(1, 2)
        attention_scores[:, :, ~padding_and_loss_attention_mask_blocks] = float("-inf")
        attention_scores = attention_scores.transpose(1, 2)
        return attention_scores

    def split_padding_and_loss_attention_mask_by_blocks(self, padding_and_loss_attention_mask):
        padding_and_loss_attention_mask_blocks = padding_and_loss_attention_mask.reshape(
            self.batch_size, self.num_blocks_kv, self.block_size_kv
        )
        padding_and_loss_attention_mask_blocks = padding_and_loss_attention_mask_blocks.unsqueeze(
            2
        ).expand(-1, -1, self.block_size_query, -1)
        return padding_and_loss_attention_mask_blocks

    def _find_future_and_diagonal_masks_for_global_calculations(self):
        mask_for_future_blocks = torch.arange(self.num_blocks_query).unsqueeze(
            0
        ).unsqueeze(0).to(
            self.sparse_query_block_indices.device
        ) > self.global_query_block_indices.unsqueeze(
            2
        )
        mask_for_diagonal_blocks = torch.arange(self.num_blocks_query).unsqueeze(
            0
        ).unsqueeze(0).to(
            self.sparse_query_block_indices.device
        ) == self.global_query_block_indices.unsqueeze(
            2
        )
        return mask_for_diagonal_blocks, mask_for_future_blocks

    def _calculate_attention_softmax_weights(
        self, attention_scores, permute_shape, reshape_shape, flatten_length
    ):
        attention_scores_permuted = attention_scores.permute(permute_shape)
        attention_scores_flat = attention_scores_permuted.reshape(
            self.num_heads, self.batch_size, -1, flatten_length
        )
        attention_weights_flat = F.softmax(attention_scores_flat, dim=-1)
        attention_weights_permuted = attention_weights_flat.view(
            attention_scores_permuted.shape
        )
        attention_weights = attention_weights_permuted.permute(reshape_shape)
        return attention_weights

    def _calculate_all_sparse_attention_scores_from_index(
        self, query_blocks, key_blocks, kv_index_table, padding_and_loss_attention_mask
    ):
        key_blocks_flat = key_blocks.reshape(
            self.num_heads,
            self.batch_size * self.num_blocks_kv,
            self.head_dim,
            self.block_size_kv,
        )
        kv_index_table_flat = kv_index_table.view(-1, kv_index_table.size(-1))
        key_blocks_by_index_flat = key_blocks_flat[:, kv_index_table_flat]
        key_blocks_by_index = key_blocks_by_index_flat.view(
            self.num_heads,
            self.batch_size,
            self.num_key_blocks_per_sparse_query_block,
            kv_index_table.size(2),
            self.head_dim,
            self.block_size_kv,
        )
        attention_scores = torch.matmul(
            query_blocks.unsqueeze(2), key_blocks_by_index
        ) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if self.is_causal_masking:
            mask_for_diagonal_blocks, mask_for_future_blocks = (
                self._find_future_and_diagonal_masks_for_sparse_calculations(
                    kv_index_table
                )
            )
            attention_scores = self._apply_causal_masking(
                attention_scores,
                mask_for_diagonal_blocks,
                mask_for_future_blocks,
                self.num_key_blocks_per_sparse_query_block,
                self.num_sparse_query_blocks,
            )
        if padding_and_loss_attention_mask != None:
            padding_and_loss_attention_mask_blocks = self.split_padding_and_loss_attention_mask_by_blocks(
                padding_and_loss_attention_mask
            )
            self._apply_padding_and_loss_attention_mask_to_sparse_attention_scores(
                attention_scores, kv_index_table_flat, padding_and_loss_attention_mask_blocks
            )
        return attention_scores

    def _apply_padding_and_loss_attention_mask_to_sparse_attention_scores(
        self, attention_scores, kv_index_table_flat, padding_and_loss_attention_mask_blocks
    ):
        padding_and_loss_attention_mask_blocks_flat = padding_and_loss_attention_mask_blocks.reshape(
            -1, self.block_size_query, self.block_size_kv
        )
        padding_and_loss_attention_mask_blocks_by_index_flat = (
            padding_and_loss_attention_mask_blocks_flat[kv_index_table_flat]
        )
        padding_and_loss_attention_mask_blocks_by_index = (
            padding_and_loss_attention_mask_blocks_by_index_flat.view(
                self.batch_size,
                self.num_key_blocks_per_sparse_query_block,
                self.num_sparse_query_blocks,
                self.block_size_query,
                self.block_size_kv,
            )
        )
        attention_scores[:, ~padding_and_loss_attention_mask_blocks_by_index] = float("-inf")

    def _find_future_and_diagonal_masks_for_sparse_calculations(self, kv_index_table):
        sparse_query_block_indices_offset_by_batch = (
            self._offset_indices_by_batch_for_direct_indexing(
                self.sparse_query_block_indices
            )
        )
        mask_for_future_blocks = (
            kv_index_table > sparse_query_block_indices_offset_by_batch.unsqueeze(1)
        )
        mask_for_diagonal_blocks = (
            kv_index_table == sparse_query_block_indices_offset_by_batch.unsqueeze(1)
        )
        return mask_for_diagonal_blocks, mask_for_future_blocks

    def _apply_causal_masking(
        self,
        attention_scores,
        mask_for_diagonal_blocks,
        mask_for_future_blocks,
        num_blocks1,
        num_blocks2,
    ):
        mask_for_future_blocks_flat = mask_for_future_blocks.view(
            -1, mask_for_future_blocks.size(-1)
        )
        mask_for_diagonal_blocks_flat = mask_for_diagonal_blocks.view(
            -1, mask_for_diagonal_blocks.size(-1)
        )
        attention_scores_flat = attention_scores.reshape(
            self.num_heads,
            self.batch_size * num_blocks1,
            num_blocks2,
            self.block_size_query,
            self.block_size_kv,
        )
        attention_scores_flat[:, mask_for_future_blocks_flat] = (
            self.pad_attention_square
        )
        diagonal_blocks = attention_scores_flat[:, mask_for_diagonal_blocks_flat]
        diagonal_blocks = diagonal_blocks.masked_fill(
            self.causal_attention_square_mask == 0, float("-inf")
        )
        attention_scores_flat[:, mask_for_diagonal_blocks_flat] = diagonal_blocks
        attention_scores = attention_scores_flat.reshape(
            self.num_heads,
            self.batch_size,
            num_blocks1,
            num_blocks2,
            self.block_size_query,
            self.block_size_kv,
        )
        return attention_scores

    def _create_kv_index_table_for_local_global_random_block_selection(self):
        local_window_kv_index_table = self._make_local_window_indices()
        random_kv_index_table = self._make_random_indices()
        global_kv_index_tables = self._make_global_kv_index_tables()
        kv_index_table = torch.cat(
            [
                local_window_kv_index_table,
                global_kv_index_tables,
                random_kv_index_table,
            ],
            dim=1,
        )
        kv_index_table = self._offset_indices_by_batch_for_direct_indexing(
            kv_index_table
        )
        return kv_index_table

    def _offset_indices_by_batch_for_direct_indexing(self, index_table):
        add_vals = torch.arange(
            0, self.batch_size * self.num_blocks_kv, self.num_blocks_kv
        ).to(index_table.device)
        add_vals = add_vals.view(*([-1] + [1] * (index_table.ndim - 1)))
        index_table = index_table + add_vals
        return index_table

    # SOMEDAY/MAYBE: Have the window shifts skip over globals (instead of automatically repeating them)
    def _make_local_window_indices(self):
        shifts = torch.arange(
            -self.local_window_extension_length, self.local_window_extension_length + 1
        ).to(self.sparse_query_block_indices.device)
        kv_index_table = (
            self.sparse_query_block_indices.unsqueeze(1)
            + shifts.unsqueeze(0).unsqueeze(-1)
        ) % self.num_blocks_kv
        return kv_index_table

    def _make_random_indices(self):
        random_kv_index_table = torch.randint(
            0,
            self.num_blocks_kv,
            (self.batch_size, self.num_random_blocks, self.num_sparse_query_blocks),
        ).to(self.sparse_query_block_indices.device)
        return random_kv_index_table

    def _make_global_kv_index_tables(self):
        return self.global_kv_block_indices.unsqueeze(-1).expand(
            -1, -1, self.num_sparse_query_blocks
        )


def _determine_first_occurrences(kv_index_table):
    sorted_kv_index_table, indices = torch.sort(kv_index_table, dim=1)
    first_occurrences_mask = torch.ones_like(sorted_kv_index_table, dtype=torch.bool)
    first_occurrences_mask[:, 1:] = (
        sorted_kv_index_table[:, 1:] != sorted_kv_index_table[:, :-1]
    )
    return sorted_kv_index_table, first_occurrences_mask


def _pad_sequence_with_duplicates(tensors):
    max_length = max(tensor.size(0) for tensor in tensors)
    padded_tensor = torch.zeros(len(tensors), max_length, dtype=torch.long)
    for i, tensor in enumerate(tensors):
        num_padding = max_length - tensor.size(0)
        if num_padding > 0:
            last_index = tensor.size(0) - 1
            padded_tensor[i, : last_index + 1] = tensor
            padded_tensor[i, last_index + 1 :] = tensor[last_index].item()
        else:
            padded_tensor[i] = tensor
    return padded_tensor
