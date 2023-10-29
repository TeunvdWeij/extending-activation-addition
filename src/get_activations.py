import os
import torch
from time import perf_counter

from model import Llama2Helper
from utils import load_pile, get_hf_token
from activation_tensor import ActivationTensor

# set variables
mode = "only_code"
num_samples = 100
model_name = "meta-llama/Llama-2-7b-chat-hf"
layer = 29
file_path = f"data/activations/{model_name.split('/')[1]}_{mode.replace('_', '-')}_no-mean_v2.09.pt"
max_seq_length = 4096
truncation = True
mean = False
note = "switching to bfloat"

assert not os.path.isfile(file_path), "File already exists, can't overwrite file."

dataset = load_pile(mode=mode, shuffle=True, split="train", iterable=True)
model = Llama2Helper(
    model_name=model_name, hf_token=get_hf_token(), dtype=torch.bfloat16
)

start_time = perf_counter()
if mean:
    avg_acts = torch.tensor(0)
else: 
    acts_list = []
total_tokens = 0

for i in range(num_samples):
    if i % (int(num_samples*0.1)) == 0:
        print(f"Iter {i} of {num_samples}, {round(i/num_samples*100, 0)}%", flush=True)

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
    
    if mean:
        avg_acts = (avg_acts * i + acts) / (i + 1)
    else:
        acts_list.append(acts)

    total_tokens += encoded.numel()

total_time = perf_counter() - start_time

if mean:
    acts_tensor = avg_acts
else:
    acts_tensor = torch.vstack(acts_list)

acts_obj = ActivationTensor(
    tensor=acts_tensor,
    mode=mode,
    num_samples=num_samples,
    model_name=model_name,
    layer=layer,
    file_path=file_path,
    max_seq_length=max_seq_length,
    truncation=truncation,
    total_time=total_time,
    total_tokens=total_tokens,
    mean=mean,
    note=note,
)

acts_obj.save()
