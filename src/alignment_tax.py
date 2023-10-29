import torch
import json
import time
import os

from model import Llama2Helper
from utils import (
    load_pile,
    get_hf_token,
    get_skip_tokens,
    acc,
)

# first check if no file is being overwritten
file_path = "results/alignment_tax_v2.05.json"
assert not os.path.isfile(file_path), "File already exists, nothing changed."

# I can change this to 1_000_000 but this does require significant compute
total_tokens_per_ic = 100_000
layer = 29
max_seq_length = 4096
injection_coefficients = (0, 20, 40, 75, 100, 150, 200, 250, 300, 400, 500)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = Llama2Helper(model_name=model_name, hf_token=get_hf_token(), dtype=torch.bfloat16)

pos_act_file_path = "data/activations/Llama-2-7b-chat-hf_only-text_2.04.pt"
pos_avg_acts = torch.load(pos_act_file_path, map_location=device).tensor
# turn the activations into a unit vector for easier scaling
pos_acts = pos_avg_acts / torch.norm(pos_avg_acts, p=2)

neg_act_file_path = "data/activations/Llama-2-7b-chat-hf_only-code_v2.08.pt"
neg_avg_acts = torch.load(neg_act_file_path, map_location=device).tensor
# turn the activations into a unit vector for easier scaling
neg_acts = neg_avg_acts / torch.norm(neg_avg_acts, p=2)

mean_act_file_path = "data/activations/Llama-2-7b-chat-hf_all_v2.07.pt"
mean_avg_acts = torch.load(neg_act_file_path, map_location=device).tensor
# turn the activations into a unit vector for easier scaling
mean_acts = mean_avg_acts / torch.norm(mean_avg_acts, p=2)

# # some dummy input to get the shape of layer
# model.get_logits(torch.tensor([[1]]))
# acts_shape = model.get_last_activations(layer).shape
# random_acts = torch.rand(acts_shape).to(torch.half).to(device)
# acts = random_acts / torch.norm(random_acts, p=2)

acts = -neg_acts 
# acts = pos_acts - neg_acts - mean_acts

results = {}
results["meta"] = {
    "model_name": model_name,
    # "act_file_path": act_file_path,
    "layer": layer,
    "max_seq_length": max_seq_length,
    "injection_coefficients": injection_coefficients,
    "total_tokens_per_ic": total_tokens_per_ic,
    "note": "subtracting code",
}

for mode in ("only_text", "only_code"):
    results[mode] = {}
    skip_tokens = get_skip_tokens(mode=mode, skip="skip50", data_type="tokens_int")
    skip_tokens = torch.tensor(skip_tokens).to(device)
    dataset = load_pile(split="validation", mode=mode, iterable=True)

    for ic in injection_coefficients:
        print(f"Mode: {mode}.   Injection Coefficient: {ic}")

        # clear and set the activations to be used in the forward pass
        model.reset_all()
        model.set_add_activations(layer, ic*acts)
        

        # init dict for this injection coefficient
        ic_res = {
            "top1_acc": [],
            "top10_acc": [],
            "skip50_top1_acc": [],
            "skip50_top10_acc": [],
            "total_encoded_tokens": [],
            "total_tokens_with_skip": [],
            "total_time_in_sec": [],
        }

        analyzed_tokens = 0
        for sample in dataset:
            if analyzed_tokens > total_tokens_per_ic:
                break
            # could use this if OOM memory issues happen
            # torch.cuda.empty_cache()
            start_time = time.perf_counter()

            # truncate to context window, pad to longest sequence. detach and to device for gpu memory usage
            encoded = (
                model.tokenizer.encode(
                    sample.get("text"),  # type: ignore
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                )
                .detach() # type: ignore
                .to(device)
            )
            predictions = model.get_logits(encoded).detach().to(device)

            # align predictions: the first token is not predicted by the model
            # and the last prediction is not encoded
            encoded = encoded[:, 1:]
            predictions = predictions[:, :-1]

            top1_preds = torch.topk(predictions, k=1, dim=-1).indices.to(device)
            top10_preds = torch.topk(predictions, k=10, dim=-1).indices.to(device)
            top1_acc = acc(encoded, top1_preds)
            top10_acc = acc(encoded, top10_preds, top1=False)

            # create filter which also checks whether true tokens are in skip50
            f_50 = ~(encoded.unsqueeze(-1) == skip_tokens).any(-1)
            total_tokens_with_skip = torch.sum(f_50).item()
            # after skipping it could leave sample with 0 tokens.
            # in this case, set acc to 0 for easier handling.
            # note that the *weighted* acc is later calculated, so 
            # this has no effect on the overall accuracy
            
            if total_tokens_with_skip > 0:
                skip50_top1_acc = acc(encoded, top1_preds, f_50)
                skip50_top10_acc = acc(encoded, top10_preds, f_50, top1=False)
            else:
                skip50_top1_acc, skip50_top10_acc = 0, 0
            
            ic_res["top1_acc"].append(top1_acc)
            ic_res["top10_acc"].append(top10_acc)
            ic_res["skip50_top1_acc"].append(skip50_top1_acc)
            ic_res["skip50_top10_acc"].append(skip50_top10_acc)
            ic_res["total_encoded_tokens"].append(encoded.numel())
            ic_res["total_tokens_with_skip"].append(total_tokens_with_skip)
            ic_res["total_time_in_sec"].append(
                round(time.perf_counter() - start_time, 3)
            )

            analyzed_tokens += encoded.numel()

        results[mode][f"injection_coefficients_{ic}"] = ic_res

with open(file_path, "w") as f:
    json.dump(results, f, indent=2)
    print(f"Written to json file succesfully!")
