import os
import torch
from time import perf_counter

from model import Llama2Helper
from utils import load_pile, get_hf_token
from activation_tensor import ActivationTensor

# set variables
mode = "only_code"
num_samples = 5000
model_name = "meta-llama/Llama-2-7b-chat-hf"
layer = 29
file_path = f"data/activations/{model_name.split('/')[1]}_{mode.replace('_', '-')}_v2.08.pt"
max_seq_length = 4096
truncation = True
note = "switching to bfloat"

assert not os.path.isfile(file_path), "File already exists, nothing changed."

dataset = load_pile(mode=mode, shuffle=True, split="train", iterable=True)
model = Llama2Helper(
    model_name=model_name, hf_token=get_hf_token(), dtype=torch.bfloat16
)

start_time = perf_counter()
avg_acts = torch.tensor(0)
total_tokens = 0

for i in range(num_samples):
    if i % 100 == 0:
        print(f"Iter {i} of {num_samples}, {round(i/num_samples*100, 0)}%", flush=True)
    # in case of OOM errors
    # torch.cuda.empty_cache()
    sample = next(dataset)["text"] # type: ignore
    encoded = model.tokenizer.encode(
        sample,
        return_tensors="pt",
        truncation=truncation,
        max_length=max_seq_length,
    )
    # forward pass to get new activations
    model.get_logits(encoded)
    # get last token's activations as these likely contain most information
    acts = model.get_last_activations(layer)[:, -1, :]
    if torch.isinf(acts).any():
        print(f"INF in activations! Iteration {i}")
        print(sample)
    
    avg_acts = (avg_acts * i + acts) / (i + 1)
    total_tokens += encoded.numel()

total_time = perf_counter() - start_time

avg_acts_obj = ActivationTensor(
    tensor=avg_acts,
    mode=mode,
    num_samples=num_samples,
    model_name=model_name,
    layer=layer,
    file_path=file_path,
    max_seq_length=max_seq_length,
    truncation=truncation,
    total_time=total_time,
    total_tokens=total_tokens,
    note=note,
)

avg_acts_obj.save()
