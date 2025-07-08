import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from attention_smithy.generators import GeneratorContext

def generate_using_pretrained_model(method):
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.forward_decode = model.forward
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    start_token = tokenizer.bos_token_id
    end_token = tokenizer.eos_token_id
    input_text = f"My purpose in life is "
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    generator = GeneratorContext(method=method, no_repeat_ngram_size=5)
    generated_batch_tensor = generator.generate_sequence(model,
                                                          end_token,
                                                          input_ids,
                                                          )
    '''
    # To visualize the outputs, if desired
    if len(generated_batch_tensor.shape) > 1:
        generated_batch_tensor = generated_batch_tensor[0]
    output_text = tokenizer.decode(generated_batch_tensor, skip_special_tokens=True)
    print()
    print(output_text)
    beam_output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    output_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    print(output_text)
    #'''