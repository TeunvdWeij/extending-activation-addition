import json
import subprocess
import torch
import random

from torch.nn.functional import log_softmax
from datasets import load_dataset  # type: ignore


def get_model_name(params: str, chat: bool):
    if chat:
        return f"meta-llama/Llama-2-{params}-hf"
    else:
        return f"meta-llama/Llama-2-{params}-chat-hf"


def load_acts(device="cuda", pos_path=None, neg_path=None):
    assert pos_path or neg_path, "No path has been passed."

    returns = []
    note = ""

    if pos_path:
        pos_acts = torch.load(pos_path, map_location=device)
        note += f"Pos acts has mode {pos_acts.mode}"
        returns.append(pos_acts)
    if neg_path:
        neg_acts = torch.load(neg_path, map_location=device)
        note += f"Neg acts has mode {neg_acts.mode}"
        returns.append(neg_acts)

    returns.append(note)
    return returns


def get_skip_tokens(mode="all", skip="skip50", data_type="tokens_int"):
    """
    Opens the skip tokens json file and returns a tensor from the specified elements.

    Arguments:
    mode (str): Must be one of ("all", "only_text", "only_code"), is about subset of data
    skip (str): Must be one of ("skip50", "skip10"), is about number of tokens
    data_type (str):  Must be one of ("tokens_int", "tokens_str"), return either ints or strings

    Returns:
    list with skipped tokens
    """

    with open("data/skip_tokens/skip_tokens.json", "r") as f:
        skip_tokens = json.load(f)

    return skip_tokens[mode][skip][data_type]


def get_hf_token():
    """
    Get hugging face token, enter your own token in this place.
    """
    with open("private_information/hf_token.txt", "r") as f:
        return f.read()


def check_gpu_memory():
    """
    Checks the available free memory on each GPU.
    """
    output = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"]
        )
        .decode()
        .strip()
        .split("\n")
    )

    for i, out in enumerate(output):
        print(f"GPU {i}: {int(out)} MB free")


def acc(t1, t2, f=None, top1=True):
    """
    This function calculates the accuracy between two tensors (t1, t2), optionally after filtering.

    Arguments:
        t1 (torch.Tensor): The first tensor to compare.
        t2 (torch.Tensor): The second tensor to compare. Same shape as t1 for 'top1' mode.
        f (None or int): The filter to optionally apply
        top1 (bool): The mode of comparison. If True, t1 and t2 are directly compared. If False,
                     t1 will add a dimension and check if any value matches with t2.

    Returns:
        float: The mean comparison result, rounded to 5 decimal places. Representing the accuracy between t1 and t2.
    """
    if f is not None:
        t1, t2 = t1[f], t2[f]

    if top1:
        comparison = t1 == t2.squeeze()
    else:
        # add another dim to see if any value matches with t2
        comparison = (t1.unsqueeze(-1) == t2).any(-1)

    return comparison.float().mean().item()


# adapted from https://github.com/pesvut/separability/blob/b435310c5728dcfacb0312799d98ba6e7146507c/src/separability/texts.py#L3
def load_data(split: str, mode: str, shuffle: bool = True, iterable: bool = True):
    """
    Load pile given certain selection.

    :param split: The split can be "train", "validation", or "test"
    :param mode: The mode which can be one of "all", "only_text", "only_code"

    :return dataset: The requested dataset based on the selection
    """

    assert mode in ("all", "only_text", "only_code", "only_python")
    assert split in ("train", "validation"), "new split type not implemented."

    if mode == "only_python":
        # this implementation is correct, the datasets are just weirdly structured
        dataset_name = "codeparrot/codeparrot-clean"
        if split == "validation":
            dataset_name += "-valid"
        print(f"dataset_name{dataset_name}")
        dataset = load_dataset(dataset_name, streaming=True, split="train")
    else:
        dataset = load_dataset(
            "monology/pile-uncopyrighted", streaming=True, split=split
        )

    # with this shuffle seed I ensure that the dataset is the same across runs
    if shuffle:
        dataset = dataset.shuffle(seed=13)

    if mode == "only_text":
        dataset = dataset.filter(
            lambda sample: sample["meta"]["pile_set_name"] != "Github"
        )
    if mode == "only_code":
        dataset = dataset.filter(
            lambda sample: sample["meta"]["pile_set_name"] == "Github"
        )
    return iter(dataset) if iterable else dataset


def get_subset_from_dataset(dataset, num_samples, mode):
    """
    Get a specific number of samples from the  dataset.

    :param dataset: The iterable dataset to sample from
    :param num_samples: The number of samples to extract

    :return sampled_text: A list of 'num_samples' text strings from the dataset
    """
    if mode == "only_python":
        key = "content"
    else:
        key = "text"
    return [next(dataset)[key] for _ in range(num_samples)]


def encode(model, text):
    encoded = model.tokenizer(
        text, truncation=True, max_length=4096, return_tensors="pt"
    )
    return encoded["input_ids"].detach().to(model.device)


def load_multi_steering_data():
    random.seed(13)
    dataset_names = [
        "myopic",
        "wealth_seeking",
        "sycophancy",
        "agreeableness",
        "anti_immigration",
    ]

    datasets = {}
    for name in dataset_names:
        with open(f"data/datasets/{name}.json", "r") as f:
            dataset = json.load(f)
            random.shuffle(dataset)
            datasets[name] = dataset

    return datasets


def load_multi_steering_acts(names=None):
    if names is None:
        names = [
            "myopic",
            "wealth_seeking",
            "sycophancy",
            "agreeableness",
            "anti_immigration",
        ]
    acts = {}
    for name in names:
        acts[name] = torch.load(f"data/activations/Llama-2-7b/multi_steering/{name}.pt")

    return acts
