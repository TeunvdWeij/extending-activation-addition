import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_token = "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"
model_name = "meta-llama/Llama-2-7b-hf"
device = "cuda"

dataset = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    device_map="auto",
    token=hf_token,
    torch_dtype=torch.half,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    token=hf_token,
    torch_dtype=torch.half,
)

top1_accs = []
for i, sample in enumerate(dataset):
    # get tokens with max length
    encoded = tokenizer(
        sample["text"], truncation=True, max_length=4096, return_tensors="pt"
    )["input_ids"].to(device)

    with torch.no_grad():
        predictions = model(encoded).logits.detach().to(device)

    # get max k=1 predictions, and squeeze to (batch_size, tokens)
    top1_preds = torch.topk(predictions, k=1, dim=-1).indices.squeeze()
    # calculate accuracy with sum (all true values) divided by total tokens. `item()` gets the float
    top1_accs.append((torch.sum(top1_preds[:, :-1] == encoded[:, 1:]) / encoded.shape[1]).item())

    if i >= 100:
        break

print(f"\nAverage accuracy in %: {round(sum(top1_accs) / len(top1_accs) * 100, 2)}")

print(f"\n\nThis is top1_acc: {top1_accs}")
