import sys
import os
import torch
import pytorch_lightning as pl
from model_import import MachineTranslationModel
from data_import import MachineTranslationDataModule
from attention_smithy.utils import seed_everything
from attention_smithy.generators import GeneratorContext
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_training_job():
    seed_everything(0)
    torch.set_float32_matmul_precision('medium')

    batch_size = 128

    data_module = MachineTranslationDataModule(
        en_filepath_suffix='_en.txt',
        de_filepath_suffix='_de.txt',
        maximum_length=100,
        batch_size=batch_size,
        num_training_samples=10000,
    )
    data_module.setup()

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[TranslationCallback()],
        strategy='auto',
        accelerator='auto',
        devices=1
    )

    model = MachineTranslationModel(
        src_vocab_size=data_module.de_vocab_size,
        tgt_vocab_size=data_module.en_vocab_size,
        tgt_padding_token=data_module.en_pad_token,
    )

    trainer.fit(model, data_module)
    torch.save(model, 'model.pth')

class TranslationCallback(pl.Callback):
    def __init__(self):
        self.generator = GeneratorContext(method='beam')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def on_train_epoch_end(self, trainer, pl_module, **kwargs):
        batch = next(iter(trainer.val_dataloaders))
        batch = tuple(x.to(pl_module.device) for x in batch)
        src_input_tensor, tgt_input_tensor, expected_output_tensor, src_padding_mask, _ = batch
        start_token = self.tokenizer.cls_token_id
        end_token = self.tokenizer.sep_token_id

        with torch.no_grad():
            src_encoded = pl_module.forward_encode(src_input_tensor, src_padding_mask)
            tgt_starting_input = torch.full((src_input_tensor.shape[0], 1), start_token, device=pl_module.device)
            generated_batch_tensor = self.generator.generate_sequence(pl_module,
                                                                      end_token,
                                                                      tgt_starting_input,
                                                                      src_encoded=src_encoded,
                                                                      src_padding_mask=src_padding_mask,
                                                                      tgt_padding_mask=None,
                                                                      )

        translation_limit = 50
        reference_translations = [self.tokenizer.decode(t, skip_special_tokens=True) for t in expected_output_tensor][:translation_limit]
        output_translations = [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_batch_tensor][:translation_limit]

        for ref, out in zip(reference_translations, output_translations):
            print(f"Reference: {ref}")
            print(f"Output: {out}\n")


if __name__ == "__main__":
    run_training_job()
