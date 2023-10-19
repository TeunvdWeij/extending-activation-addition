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
    encoded = tokenizer(
        sample["text"], truncation=True, max_length=4096, return_tensors="pt"
    )["input_ids"].to(device)

    predictions = model.get_logits(encoded).detach().to(device)

    top1_preds = torch.topk(predictions, k=1, dim=-1).indices.to(device)
    top1_accs = torch.sum(top1_preds == encoded) / encoded.shape[1]

    if i >= 10:
        break

print(f"This is top1_acc: {top1_accs}")
print(f"Average accuracy in %: {round(sum(top1_accs) / len(top1_accs) * 100, 2)}")
