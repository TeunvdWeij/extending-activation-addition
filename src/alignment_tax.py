import torch
import json
import time
import os

from model import Llama2Helper
from utils import load_pile, get_subset_from_dataset, get_hf_token, get_skip_tokens, acc, check_gpu_memory

# first check if no file is being overwritten
file_path = "results/alignment_tax_v0.3.json"
assert not os.path.isfile(file_path), "File already exists, nothing changed."


model_name = "meta-llama/Llama-2-13b-hf"
model = Llama2Helper(model_name=model_name, hf_token=get_hf_token())

total_tokens = 100_000
# batch size needs to be this small for memory reasons
batch_size = 2
mode = "all"
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cuda:1" if torch.cuda.is_available() else "cpu"
# print(f"device is: {device}")

skip_tokens = get_skip_tokens(mode=mode, skip="skip50", data_type="tokens_int")
skip_tokens = torch.tensor(skip_tokens).to(device)

dataset = load_pile(split="train", mode=mode, batch_size=batch_size)

results = {}

analyzed_tokens = 0
batch_num = 0
while analyzed_tokens < total_tokens:
    torch.cuda.empty_cache()
    print(f"Batch {batch_num}, {round(analyzed_tokens/total_tokens*100, 1)}% of total tokens")
    check_gpu_memory()
    start_time = time.perf_counter()
    # could use this if OOM memory issues happen
    # torch.cuda.empty_cache()
    ds_subset = get_subset_from_dataset(dataset, batch_size)

    # truncate to context window, pad to longest sequence. detach and to device for gpu memory usage
    encoded = model.tokenizer(ds_subset, truncation=True, max_length=4096, return_tensors="pt", padding="longest")["input_ids"].detach().to(device)
    print(f"encoded shape: {encoded.shape}, mem: {encoded.element_size() * encoded.nelement()}")

    # create filter (f) which checks if token is not padding
    #NOTE: this does mean that we never assess whether </s> is predicted correctly
    f = ~(encoded == model.tokenizer.pad_token_id)
    # create filter which also checks whether true tokens are in skip50 
    f_50 = ~(f.unsqueeze(-1) == skip_tokens).any(-1)
    
    preds = model.get_logits(encoded).detach().to(device)
    
    # squeeze does: (batch_size, max_token_length, 1) -->  (batch_size, max_token_length)
    top1_preds = torch.topk(preds, k=1, dim=-1).indices.squeeze().to(device)
    top10_preds = torch.topk(preds, k=10, dim=-1).indices.to(device)

    top1_acc = acc(encoded, top1_preds, f)
    top10_acc = acc(encoded, top10_preds, f, top1=False)
    
    skip50_top1_acc = acc(encoded, top1_preds, f_50)
    skip50_top10_acc = acc(encoded, top10_preds, f_50, top1=False)
    
    results[f"batch_{batch_num}"] = {
        "top1_acc": top1_acc,
        "top10_acc": top10_acc,
        "skip50_top1_acc": skip50_top1_acc,
        "skip50_top10_acc": skip50_top10_acc,
        "total_encoded_tokens": encoded.numel(),
        "total_pad_token_ids": torch.sum(f).item(),
        "total_skipped_tokens": torch.sum(f_50).item(),
        "total_time_in_sec": round(time.perf_counter() - start_time, 3)
        }
    
    batch_num += 1
    analyzed_tokens += torch.sum(f).item()

    # temporary safety measure
    if batch_num > 100:
        break

with open(file_path, "w") as f:
    json.dump(results, f, indent=2)
    print(f"Written to json file succesfully for {batch_num} batches!")

