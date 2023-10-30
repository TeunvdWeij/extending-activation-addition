import os
import torch
import argparse

from time import perf_counter

from model import Llama2Helper
from utils import load_pile, get_hf_token, get_model_name
from activation_tensor import ActivationTensor


def get_activations(acts_obj: ActivationTensor):
    dataset = load_pile(mode=acts_obj.mode, shuffle=True, split="train", iterable=True)
    model = Llama2Helper(
        model_name=acts_obj.model_name, hf_token=get_hf_token(), dtype=acts_obj.dtype
    )

    total_tokens = 0
    start_time = perf_counter()
    for i in range(acts_obj.num_samples):
        # print progress every ten percent
        if i % (int(acts_obj.num_samples * 0.1)) == 0:
            print(
                f"Iter {i} of {acts_obj.num_samples}  Time passed: {round(perf_counter() -start_time)} sec",
                flush=True,
            )
        # torch.cuda.empty_cache()
        sample = next(dataset)["text"]  # type: ignore
        encoded = model.tokenizer.encode(
            sample,
            return_tensors="pt",
            truncation=acts_obj.truncation,
            max_length=acts_obj.max_seq_length,
        )
        # forward pass to get new activations
        model.get_logits(encoded)
        # get last token's activations as these likely contain most information
        acts = model.get_last_activations(acts_obj.layer)[:, -1, :].detach().cpu()

        acts_obj.process_new_acts(acts, i)

        total_tokens += encoded.numel()  # type: ignore

    total_time = perf_counter() - start_time

    acts_obj.save()


def arg_parser():
    parser = argparse.ArgumentParser()
    # required params
    parser.add_argument("--note", type=str)
    parser.add_argument("--version", type=str)

    parser.add_argument("--num_samples", type=int, default=5000)
    # change so multiple layers can be used
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--max_seq_length", type=int, default=4096)
        
    parser.add_argument('--chat', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--truncation', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--mean', default=True, action=argparse.BooleanOptionalAction)
        
    parser.add_argument(
        "--mode",
        type=str,
        choices=["only_code", "only_text", "all"],
        default="only_code",
    )
    parser.add_argument(
        "--model_params", type=str, choices=["7b", "13b", "70b"], default="7b"
    )

    parser.add_argument(
        "--dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16"
    )
    return parser.parse_args()


def main():
    args = arg_parser()

    acts_obj = ActivationTensor(
        args.note,
        args.version,
        args.num_samples,
        args.mode,
        args.model_params,
        args.chat,
        args.layer,
        args.max_seq_length,
        args.truncation,
        args.mean,
        args.dtype,
    )

    print(f"Acts object: {acts_obj.__dict__}")
    get_activations(acts_obj)


if __name__ == "__main__":
    main()
