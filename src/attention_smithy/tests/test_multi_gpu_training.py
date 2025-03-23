import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from attention_smithy.components import (
    Encoder, Decoder, EncoderLayer, DecoderLayer,
    MultiheadAttention, FeedForwardNetwork
)
from attention_smithy.numeric_embeddings import (
    SinusoidalPositionEmbedding, LearnedPositionEmbedding,
    RotaryPositionEmbedding, ALiBiPositionEmbedding,
    NumericEmbeddingManager
)
from attention_smithy.attention import StandardAttentionMethod


# Move callback class outside of the test function
class GradientCheckCallback(pl.Callback):
    def on_after_backward(self, trainer, model):
        for name, param in model.named_parameters():
            if 'rotary_position' in name or 'freqs' in name:
                continue
            if param.requires_grad and param.grad is None:
                print(f"Parameter {name} has no gradient after backward pass")


class MiniTransformerForTesting(pl.LightningModule):
    def __init__(self, src_vocab_size=100, tgt_vocab_size=100):
        super().__init__()
        self.embedding_dimension = 32
        self.num_heads = 2

        self.src_embed = torch.nn.Embedding(src_vocab_size, self.embedding_dimension)
        self.tgt_embed = torch.nn.Embedding(tgt_vocab_size, self.embedding_dimension)

        # Position embeddings
        self.numeric_embedding_manager = NumericEmbeddingManager(
            sinusoidal_position=SinusoidalPositionEmbedding(self.embedding_dimension),
            learned_position=LearnedPositionEmbedding(max_sequence_length=50,
                                                    embedding_dimension=self.embedding_dimension),
            rotary_position=RotaryPositionEmbedding(self.embedding_dimension // self.num_heads),
            alibi_position=ALiBiPositionEmbedding(self.num_heads)
        )

        # Attention components
        generic_attention = MultiheadAttention(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.num_heads,
            attention_method=StandardAttentionMethod(dropout=0.1)
        )

        decoder_self_attention = MultiheadAttention(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.num_heads,
            attention_method=StandardAttentionMethod(dropout=0.1, is_causal_masking=True)
        )

        # FFN
        ffn = FeedForwardNetwork(
            self.embedding_dimension,
            feed_forward_dimension=128,
            activation_function_string='relu',
            dropout=0.1
        )

        # Encoder & Decoder
        encoder_layer = EncoderLayer(
            self.embedding_dimension,
            generic_attention,
            ffn,
            dropout=0.1
        )
        self.encoder = Encoder(encoder_layer, number_of_layers=1)

        decoder_layer = DecoderLayer(
            self.embedding_dimension,
            decoder_self_attention,
            generic_attention,
            ffn,
            dropout=0.1
        )
        self.decoder = Decoder(decoder_layer, number_of_layers=1)

        self.output_layer = torch.nn.Linear(self.embedding_dimension, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_embedding = self.src_embed(src) * (self.embedding_dimension ** 0.5)
        tgt_embedding = self.tgt_embed(tgt) * (self.embedding_dimension ** 0.5)

        # Add position embeddings
        src_pos = self.numeric_embedding_manager.calculate_sinusoidal_and_learned_tokenizations(src_embedding)

        # Encode and decode
        encoded = self.encoder(
            src=src_embedding + src_pos,
            src_padding_mask=src_mask,
            numeric_embedding_manager=self.numeric_embedding_manager
        )

        decoded = self.decoder(
            tgt=tgt_embedding,
            src=encoded,
            tgt_padding_mask=tgt_mask,
            src_padding_mask=src_mask,
            numeric_embedding_manager=self.numeric_embedding_manager
        )

        return self.output_layer(decoded)

    def training_step(self, batch, batch_idx):
        src, tgt, src_mask, tgt_mask = batch
        output = self(src, tgt, src_mask, tgt_mask)

        # Reshape output and target for loss computation
        output_flat = output.view(-1, output.size(-1))
        target_flat = tgt.view(-1)

        # Make sure requires_grad is True
        if not output_flat.requires_grad:
            raise ValueError("Output tensor doesn't require gradients")

        loss = torch.nn.functional.cross_entropy(
            output_flat,
            target_flat,
            ignore_index=0,  # Assuming 0 is padding
            reduction='mean'
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


@pytest.fixture
def test_dataloader():
    # Create small dummy dataset
    batch_size = 4
    seq_length = 10
    num_samples = 16

    src = torch.randint(1, 100, (num_samples, seq_length))
    tgt = torch.randint(1, 100, (num_samples, seq_length))
    src_mask = (src != 0).float()
    tgt_mask = (tgt != 0).float()

    dataset = TensorDataset(src, tgt, src_mask, tgt_mask)
    return DataLoader(dataset, batch_size=batch_size)


def check_gpu_count():
    """Check if multiple GPUs are available."""
    if torch.cuda.is_available():
        return torch.cuda.device_count() > 1
    return False


@pytest.mark.skipif(not check_gpu_count(), reason="Test requires multiple GPUs")
def test_ddp_training(test_dataloader):
    model = MiniTransformerForTesting()

    # Check that learnable parameters require gradients before training
    for name, param in model.named_parameters():
        if 'rotary_position' in name or 'freqs' in name:
            continue
        assert param.requires_grad, f"Parameter {name} doesn't require gradients"

    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_progress_bar=True,
        logger=False,
        detect_anomaly=True,
    )

    trainer.callbacks.append(GradientCheckCallback())
    trainer.fit(model, test_dataloader)

    # Check gradients after training
    for name, param in model.named_parameters():
        if 'rotary_position' in name or 'freqs' in name:
            continue
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.any(param.grad != 0), f"Parameter {name} has zero gradient"


@pytest.mark.skipif(not check_gpu_count(), reason="Test requires multiple GPUs")
def test_model_save_load(test_dataloader, tmp_path):
    model = MiniTransformerForTesting()
    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_progress_bar=True,
        logger=False,
        detect_anomaly=True,
    )

    # Train and save
    trainer.fit(model, test_dataloader)

    checkpoint_path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    # Load and verify
    loaded_model = MiniTransformerForTesting.load_from_checkpoint(checkpoint_path)
    assert isinstance(loaded_model, MiniTransformerForTesting)

    # Compare model parameters
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
        assert torch.allclose(p1.data, p2.data), f"Parameters {n1} don't match"