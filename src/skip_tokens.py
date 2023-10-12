# this code generates the json file in the data fodler containing the skip10/50 tokens for the Pile dataset for Llama 2
import numpy as np
import json
import os

from model import init_tokenizer
from utils import load_pile, get_subset_from_dataset

# first check if no file is being overwritten
file_path = "data/skip_tokens.json"
assert not os.path.isfile(file_path), "File already exists, nothing changed."

with open("private_information/hf_token.txt", "r") as f:
    hf_token = f.read()

# get the tokenizer for the llama2 models
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = init_tokenizer(model_name, hf_token=hf_token)
results = {}
num_samples = 10_000


# for with and without code
for mode in ("all", "only_text", "only_code"):
    dataset = load_pile(split="train", mode=mode)
    ds_subset = get_subset_from_dataset(dataset, num_samples=num_samples)

    # encode all the text and make array 1D
    encoded = tokenizer(ds_subset, return_tensors="np")

    # get the unique tokens and their corresponding counts
    unique, counts = np.unique(np.hstack(encoded["input_ids"]), return_counts=True)

    # sort counts, get index, get top 50, to list for json
    most_frequent = unique[counts.argsort()[-50:]].tolist()

    tokens_str = [tokenizer.decode(token_int) for token_int in most_frequent]

    results[mode] = {
        "skip50": {"tokens_int": most_frequent, "tokens_str": tokens_str},
        "skip10": {"tokens_int": most_frequent[:10], "tokens_str": tokens_str[:10]},
    }

with open(file_path, "w") as f:
    json.dump(results, f, indent=2)
    print(f"Written to json file succesfully for {num_samples} samples!")
