# this code generates the json file in the data fodler containing the skip10/50 tokens for the Pile dataset for Llama 2
import numpy as np 
import json 
import os

from model import init_tokenizer
from utils import load_pile

file_path = "data/skip_tokens.json"
assert not os.path.isfile(file_path), "File already exists, nothing changed."


def get_subset_from_dataset(dataset, num_samples):
    much_text = []
    for i, batch  in enumerate(dataset.shuffle(seed=13, buffer_size=num_samples)):
        if i > num_samples:
            break
        much_text.append(batch['text'])
    return much_text

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = init_tokenizer(model_name)

dataset = load_pile(split="validation", include_code=False)
text_subset = get_subset_from_dataset(dataset, num_samples=10_000)

# encode all the text and make array 1D
encoded = tokenizer(text_subset, return_tensors="np")

# get the unique tokens and their corresponding counts
unique, counts = np.unique(np.hstack(encoded["input_ids"]), return_counts=True)

# sort counts, get index, reverse for decreasing, get top 50, to list for json
most_frequent = unique[counts.argsort()[:-51:-1]].tolist()

tokens_str = [tokenizer.decode(token_int) for token_int in most_frequent]

skip_tokens = {
    "skip50": {"tokens_int": most_frequent,
               "tokens_str": tokens_str},
    "skip10": {"tokens_int": most_frequent[:10],
               "tokens_str": tokens_str[:10]}
    }


with open(file_path, "w") as f:
    json.dump(skip_tokens, f, indent=4)