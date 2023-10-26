import os 
from tqdm import tqdm
from time import perf_counter

from model import Llama2Helper
from utils import load_pile, get_hf_token
from activation_tensor import ActivationTensor

# set variables
mode = "only_text"
num_samples = 5000
model_name = "meta-llama/Llama-2-7b-chat-hf"
layer = 29
file_path = f"data/activations/{model_name.split('/')[1]}_v2.1.pt"
max_seq_length = 4096
truncation = True

assert not os.path.isfile(file_path), "File already exists, nothing changed."

dataset = load_pile(mode=mode, split="train", iterable=True)
model = Llama2Helper(model_name=model_name, hf_token=get_hf_token())

start_time = perf_counter()
avg_acts = 0
for i in range(num_samples):
    # in case of OOM errors
    # torch.cuda.empty_cache()
    encoded = model.tokenizer.encode(
        next(dataset)["text"],
        return_tensors="pt",
        truncation=truncation,
        max_length=max_seq_length,
    )
    # forward pass to get new activations
    model.get_logits(encoded)
    # get last token's activations as these likely contain most information 
    acts = model.get_last_activations(layer)[:, -1, :]
    avg_acts = (avg_acts * i + acts) / (i+1)

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
    total_time=total_time
)

avg_acts_obj.save()

