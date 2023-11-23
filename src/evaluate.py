import torch
import time
import argparse
import numpy as np


from evaluation import Evaluation

from utils import load_pile


def evaluate(eval_obj: Evaluation):
    model = eval_obj.get_model()
    eval_obj.set_acts()

    for mode in eval_obj.modes:
        eval_obj.results[mode] = {}
        skip_tokens = eval_obj.get_skip_tokens(mode)
        dataset = load_pile(split="validation", mode=mode, iterable=True)

        for ic in eval_obj.ics:
            print(f"Mode: {mode}.   Injection Coefficient: {ic}", flush=True)
            # clear and set the activations to be used in the forward pass
            model.reset_all()
            for layer in eval_obj.layers:
                # this will only be one layer if mean is false, see evaluation.py
                model.set_add_activations(layer, ic * eval_obj.acts[layer])


            # init list for storing results for this injection coefficient
            ic_results = []
            analyzed_tokens = 0

            for sample in dataset:
                start_time = time.perf_counter()

                if analyzed_tokens > eval_obj.total_tokens_per_ic:
                    break
                # can use this if OOM memory issues happen, probably slows code down a bit
                torch.cuda.empty_cache()
                encoded = eval_obj.encode(sample)
                predictions = model.get_logits(encoded).detach().to(model.device)

                sample_results = eval_obj.process_predictions(
                    encoded,
                    predictions,
                    start_time,
                    skip_tokens,
                )

                ic_results.append(sample_results)

                analyzed_tokens += encoded.numel()

            # TODO: switch to numpy for slicing, but this can be done better
            ic_results = np.array(ic_results)
            ic_results_dict = {}
            for i, key in enumerate(eval_obj.result_keys):
                # to list for json serialization
                ic_results_dict[key] = list(ic_results[:, i])
            eval_obj.results[mode][f"injection_coefficients_{ic}"] = ic_results_dict

    eval_obj.save()


def arg_parser():
    # TODO: provide descriptions
    parser = argparse.ArgumentParser()

    # required params
    parser.add_argument("--note", type=str)
    parser.add_argument("--version", type=str)

    # default ints
    parser.add_argument("--total_tokens_per_ic", type=int, default=100_000)
    parser.add_argument("--max_seq_length", type=int, default=4096)

    # default bools
    parser.add_argument("--chat", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--truncation", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--mean", default=True, action=argparse.BooleanOptionalAction)

    # default list
    # ics means injection coefficients
    parser.add_argument(
        "--ics",
        nargs="+",
        type=int,
        default=[0, 20, 40, 75, 100, 150, 200, 250, 300, 400, 500],
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[29])

    # default str
    parser.add_argument(
        "--model_params", type=str, choices=["7b", "13b", "70b"], default="7b"
    )
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument(
        "--pos_acts",
        nargs="+",
        type=str,
        choices=["all", "only_text", "only_code", "random"],
        default="",
    )
    parser.add_argument(
        "--neg_acts",
        nargs="+",
        type=str,
        choices=["all", "only_text", "only_code", "random"],
        default="",
    )
    return parser.parse_args()


def main():
    args = arg_parser()

    eval_obj = Evaluation(
        args.note,
        args.version,
        args.total_tokens_per_ic,
        args.max_seq_length,
        args.chat,
        args.truncation,
        args.mean,
        args.ics,
        args.layers,
        args.model_params,
        args.dtype,
        args.pos_acts,
        args.neg_acts,
    )

    print(f"Eval object: {eval_obj.__dict__}")
    evaluate(eval_obj)


if __name__ == "__main__":
    main()