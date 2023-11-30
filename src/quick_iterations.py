import torch
import itertools
import einops
from time import perf_counter
from datasets import load_dataset

from model import Llama2Helper
from utils import get_hf_token

dataset = load_dataset("Anthropic/model-written-evals")

hf_token = get_hf_token()
model_name = "meta-llama/Llama-2-7b-chat-hf"
device = "cuda"

model = Llama2Helper(model_name=model_name, hf_token=hf_token)

# the correct answer is 4
q = """What is the output of the below? Only state the output, and nothing else.
```
count = 1

def doThis():
    global count
    for i in (1, 2, 3):
        count += 1

doThis()

print (count)
```
"""

# inputs = ["python count", "python def", "python for", "python in"]
# inputs = ["count", "def", "for", "in"]
inputs = ["python", "Python"]

acts_list = []
# layers = [25, 26, 27, 28]
layers = [25, 25, 25, 25]
for i, l in zip(inputs, layers):
    encoded = model.tokenizer(i, truncation=True, max_length=4096, return_tensors="pt")[
        "input_ids"
    ].to(device)
    model.get_logits(encoded)
    acts = model.get_last_activations(l)[:, -1, :]
    acts_list.append(acts)

print("Normal Performance:")
model.reset_all()
for _ in range(10):
    print(model.generate_text(q, do_sample=True)[len(q) :])

print("\n\n_______________")
combinations = itertools.combinations(acts_list, 2)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
print("Cosine similarity")
for c in combinations:
    print(cos(c[0], c[1]))
print("\n\n_______________")

mean_acts = einops.reduce(acts_list, "acts batch dim_model -> batch dim_model", "mean")
mean_acts.shape

model.reset_all()
for acts, l in zip(acts_list, layers):
    model.set_add_activations(l, -0.5 * mean_acts)
    break

print("Normal performance")

for _ in range(10):
    print(model.generate_text(q, do_sample=True)[len(q) :])

# repo = "codeparrot/github-code-clean"
repo = "codeparrot/codeparrot-clean"

ds = load_dataset(repo, "all-all", streaming=True, split="train")
ds = iter(ds)

# seqs = next(ds)
# print(seq['content'][:10_000])
print("\n\n_______________")
print("Accuracy normal")

top1_accs = []
start_time = perf_counter()
samples = [next(ds) for _ in range(100)]
model.reset_all()
for i, sample in enumerate(samples):
    #     model.set_add_activations()

    torch.cuda.empty_cache()
    encoded = model.tokenizer(
        sample["content"], truncation=True, max_length=2048, return_tensors="pt"
    )["input_ids"].to(device)
    #     print(f"Iteration: {i}")
    # print(encoded.shape)

    with torch.no_grad():
        predictions = model.get_logits(encoded).detach().to("cuda")

    # align predictions: the first token is not predicted by the model and the last prediction is not encoded
    encoded = encoded[:, 1:]
    predictions = predictions[:, :-1]

    top1_preds = torch.topk(predictions, k=1, dim=-1).indices.squeeze(-1)
    top1_accs.append((torch.sum(top1_preds == encoded) / predictions.shape[1]).item())

    if i >= 100:
        break
#
print(f"total time: {perf_counter() - start_time}")
print(f"\nThis is top1_acc: {top1_accs}")
print(
    f"\nincorrect Average accuracy in %: {round(sum(top1_accs) / len(top1_accs) * 100, 2)}"
)


print("\n\n_______________")
print("Accuracy with activations")

top1_accs = []
start_time = perf_counter()
samples = [next(ds) for _ in range(100)]
model.reset_all()
for acts, l in zip(acts_list, layers):
    model.set_add_activations(l, -0.5 * mean_acts)

for i, sample in enumerate(samples):
    #     model.set_add_activations()

    torch.cuda.empty_cache()
    encoded = model.tokenizer(
        sample["content"], truncation=True, max_length=2048, return_tensors="pt"
    )["input_ids"].to(device)
    #     print(f"Iteration: {i}")
    # print(encoded.shape)

    with torch.no_grad():
        predictions = model.get_logits(encoded).detach().to("cuda")

    # align predictions: the first token is not predicted by the model and the last prediction is not encoded
    encoded = encoded[:, 1:]
    predictions = predictions[:, :-1]

    top1_preds = torch.topk(predictions, k=1, dim=-1).indices.squeeze(-1)
    top1_accs.append((torch.sum(top1_preds == encoded) / predictions.shape[1]).item())

    if i >= 100:
        break
#
print(f"total time: {perf_counter() - start_time}")
print(f"\nThis is top1_acc: {top1_accs}")
print(
    f"\nincorrect Average accuracy in %: {round(sum(top1_accs) / len(top1_accs) * 100, 2)}"
)


# layer = 29
# lissie = ["from", "import", "return", "class", "def"]
# acts_list = []

# for i, text in enumerate(lissie):
#     torch.cuda.empty_cache()
#     encoded = model.tokenizer.encode(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=4096)

#     model.get_logits(encoded)

#     acts = model.get_last_activations(layer)[:, -1, :]
# #     avg_acts = (avg_acts * (i-1) + acts) / (i+1)
#     acts_list.append(acts)

# acts = torch.sum(torch.vstack(tuple(acts_list)), dim=0)
# acts = normalize(acts.unsqueeze(0), p=2)
# acts.shape, torch.norm(acts, dim=-1)

# # check for infs and nans, see https://stackoverflow.com/questions/48158017/pytorch-operation-to-detect-nans
# (acts == torch.inf).any(), (acts != acts).any()

# all_code = iter(load_dataset("codeparrot/github-code", streaming=True, split="train"))

# top1_accs = []
# elements = []
# start_time = perf_counter()
# # samples = [next(ds) for _ in range(10)]
# # key = "content"
# samples = [next(all_code) for _ in range(10)]
# key = "code"
# model.reset_all()
# model.set_add_activations(layer, -50*acts)

# for i, sample in enumerate(samples):

#     torch.cuda.empty_cache()
#     encoded = model.tokenizer(
#         sample[key], truncation=True, max_length=4096, return_tensors="pt"
#     )["input_ids"].to(device)
#     print(f"Iteration: {i}")
#     elements.append(encoded.numel())

#     with torch.no_grad():
#         predictions = model.get_logits(encoded).detach().to("cuda")

#     # align predictions: the first token is not predicted by the model and the last prediction is not encoded
#     encoded = encoded[:, 1:]
#     predictions = predictions[:, :-1]

#     top1_preds = torch.topk(predictions, k=1, dim=-1).indices.squeeze(-1)
#     top1_accs.append((torch.sum(top1_preds == encoded) / predictions.shape[1]).item())

#     if i >= 100:
#         break

# print(f"total time: {perf_counter() - start_time}")
# print(f"\nThis is top1_acc: {top1_accs}")

# total = sum([acc*ele for acc, ele in zip(top1_accs, elements)])
# print(f"\nAverage accuracy in %: {round(total / sum(elements) * 100, 2)}")

# sd

# This is top1_acc: [0.517241358757019, 0.6990291476249695, 0.7609660029411316, 0.8271062970161438, 0.5330459475517273, 0.8172839879989624, 0.39393940567970276, 0.7431013584136963, 0.5762711763381958, 0.4642857313156128]

# total = sum([acc*ele for acc, ele in zip(top1_accs, elements)])
# print(f"\nAverage accuracy in %: {round(total / sum(elements) * 100, 2)}")

# list(samples[0].keys())

# # all code!
# [print(i[key][:1000], "\n\n\n-----------------") for i in samples]

# [print(i[key]) for i in samples]

# from tqdm import tqdm

# all_code = iter(load_dataset("codeparrot/github-code", streaming=True, split="train"))

# N = 5000
# # samples = [next(all_code) for _ in range(N)]
# # python_lissie = []
# # other_lissie = []
# empty_all, empty_python = True, True
# i = 0
# while empty_all or empty_python:
#     i += 1
#     sample = next(all_code)
#     if sample['language'] == "Python":
#         if empty_python:
#             python_lissie.append(model.tokenizer(
#                 sample["code"], truncation=True, max_length=4096, return_tensors="pt")["input_ids"])
#             if len(python_lissie) > 10_000:
#                 empty_python = False
#     else:
#         continue
#         if empty_all:
#             other_lissie.append(model.tokenizer(
#                 sample["code"], truncation=True, max_length=4096, return_tensors="pt")["input_ids"])
#             if len(other_lissie) > 10_000:
#                 empty_all = False

#     if i % 100 == 0:
#         print(i)
#     if i > 80_000:
#         break

# # get the unique tokens and their corresponding counts
# py_unique, py_counts = torch.unique(torch.hstack(python_lissie), return_counts=True)

# # sort counts, get index, get top 50, to list for json
# py_most_frequent = py_unique[py_counts.argsort()[-50:]].tolist()

# # get the unique tokens and their corresponding counts
# other_unique, other_counts = torch.unique(torch.hstack(other_lissie), return_counts=True)

# # sort counts, get index, get top 50, to list for json
# other_most_frequent = other_unique[other_counts.argsort()[-50:]].tolist()

# # nearly equal, could still normalize
# sum(py_counts), sum(other_counts)

# py_dict = dict(zip(py_unique.tolist(), py_counts.tolist()))
# other_dict = dict(zip(other_unique.tolist(), other_counts.tolist()))

# a = list(other_dict.keys())[:10]
# b = list(py_dict.keys())[:10]

# a, b

# freqs = {}
# totals = []
# rel_freqs = []
# for key, value in py_dict.items():
#     freqs[key] = {}
#     try:
#         total = (value + other_dict[key])
#         rel_freq = round(value / total, 3)
#     except KeyError:
#         total = value
#         rel_freq = 1

#     rel_freqs.append(rel_freq)
#     totals.append(total)

#     freqs[key]['relative_frequency'] = rel_freq
#     freqs[key]['absolute_frequency'] = total

# import matplotlib.pyplot as plt

# plt.hist(rel_freqs, bins=50)
# plt.title("Frequency of python divided by (python + other)")
# plt.show()

# min(rel_freqs)

# for i, (r, t) in enumerate(zip(rel_freqs, totals)):
#     if r < 0.01:
#         print(r, t)

# #     if i > 100:
# #         break

# s
