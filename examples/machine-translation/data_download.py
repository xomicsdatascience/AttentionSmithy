from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

directory = Path("data")
directory.mkdir(parents=True, exist_ok=True)

translation_dataset = load_dataset('wmt14', 'de-en')
train_en_filepath = 'data/train_en.txt'
train_de_filepath = 'data/train_de.txt'
val_en_filepath = 'data/val_en.txt'
val_de_filepath = 'data/val_de.txt'
test_en_filepath = 'data/test_en.txt'
test_de_filepath = 'data/test_de.txt'
de_tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print('\n'*5)
print(translation_dataset.keys())

def print_tokens_to_files(data_split, de_file, en_file):
    print(f'\n\n{data_split} length: {len(translation_dataset[data_split])}')
    for i in range(len(translation_dataset[data_split])):
        if i % 100000 == 0: print(i)
        sample_de = translation_dataset[data_split][i]['translation']['de']
        sample_en = translation_dataset[data_split][i]['translation']['en']
        de_inputs = de_tokenizer(sample_de, return_tensors='pt')['input_ids'][0]
        en_inputs = en_tokenizer(sample_en, return_tensors='pt')['input_ids'][0]
        print(','.join([str(int(token)) for token in de_inputs]), file=de_file)
        print(','.join([str(int(token)) for token in en_inputs]), file=en_file)

with open(val_de_filepath, 'w') as val_de_file, open(val_en_filepath, 'w') as val_en_file:
    print_tokens_to_files("validation", val_de_file, val_en_file)

with open(train_de_filepath, 'w') as train_de_file, open(train_en_filepath, 'w') as train_en_file:
    print_tokens_to_files("train", train_de_file, train_en_file)

with open(test_de_filepath, 'w') as test_de_file, open(test_en_filepath, 'w') as test_en_file:
    print_tokens_to_files("test", test_de_file, test_en_file)
