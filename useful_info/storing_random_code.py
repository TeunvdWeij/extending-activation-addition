# how to get gpt2-xl model architecture printed

import torch
from transformers import GPT2Tokenizer, GPT2Model
name = "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(name)
model = GPT2Model.from_pretrained(name, device_map="auto", torch_dtype=torch.half)

print(model.modules)
