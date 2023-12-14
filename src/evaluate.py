import torch
import time
import argparse
import numpy as np


from evaluation import Evaluation

from utils import load_data


def evaluate(eval_obj: Evaluation):
    model = eval_obj.get_model()
    eval_obj.set_acts()
    skip_tokens_modes = []
    datasets = []

    for mode in eval_obj.modes:
        eval_obj.results[mode] = {}
        skip_tokens_modes.append(eval_obj.get_skip_tokens(mode))
        datasets.append(load_data(split="validation", mode=mode, iterable=True))

    ic_idx = 0
    used_ics = []
    default_scores = {}

    # stop when score is below certain level, see below
    keep_going = True
    while keep_going:
        try:
            ic = eval_obj.ics[ic_idx]
        except IndexError:
            break
        # store the float of all the used injection coefficients
        if eval_obj.iter_over_all_ics:
            used_ics.append(ic)
        else:
            used_ics.append(ic.item())
        print(f"\n----\nInjection Coefficient: {ic} index: {ic_idx}", flush=True)
        # clear and set the activations to be used in the forward pass
        model.reset_all()
        for layer in eval_obj.layers:
            # this will only be one layer if mean is false, see evaluation.py
            model.set_add_activations(layer, ic * eval_obj.acts[layer])

        for mode, dataset, skip_tokens in zip(
            eval_obj.modes, datasets, skip_tokens_modes
        ):
            # init list for storing results for this injection coefficient
            ic_results = []
            analyzed_tokens = 0
            # sometimes there are many results with low scores, keep track and
            # go to next if it has occured >= 5 times
            lower_than_15_percent = 0 

            for sample in dataset:
                start_time = time.perf_counter()

                if analyzed_tokens > eval_obj.total_tokens_per_ic:
                    break
                # can use this if OOM memory issues happen, probably slows code down a bit
                torch.cuda.empty_cache()
                encoded = eval_obj.encode(sample, mode)
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

            avg_top1_score = np.average(ic_results[:, 0], weights=ic_results[:, 4])
            print(f"Top1 score {mode}: {avg_top1_score}",  flush=True)
            if ic_idx == 0:
                default_scores[mode] = avg_top1_score
            # because the ic_idx is already increased with the previous mode
            elif mode == eval_obj.modes[1] and ic_idx == 1:
                    default_scores[mode] = avg_top1_score
            else:
                relative_score = avg_top1_score / default_scores[mode]
                print(f"Relative score for {mode}: {round(avg_top1_score / default_scores[mode], 5)}\n", flush=True)

            # select the next ic based on the score of the first mode
            if mode == eval_obj.modes[0]:
                # here just iterate over loop, will break when idx error arises
                if eval_obj.iter_over_all_ics:
                    ic_idx += 1
                    continue

                # choose ic_idx based on performance if ics are not manually given
                if ic_idx == 0:
                    ic_idx += 1
                    continue

                if relative_score > 0.99:
                    ic_idx += 5
                elif relative_score > 0.98:
                    ic_idx += 3
                elif relative_score < 0.05:
                    keep_going = False
                elif relative_score < 0.15:
                    lower_than_15_percent += 1
                    if lower_than_15_percent >= 5:
                        keep_going = False
                    ic_idx += 5
                else:
                    ic_idx += 1


    # save the used injection coefficients
    eval_obj.set_used_ics(used_ics)
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
    # NOTE: this is about which layers to add activations to at the same time during inference
    # it is not about running the same experiment for multiple layers in different experiments.
    parser.add_argument("--layers", nargs="+", type=int, default=[29])

    parser.add_argument("--ics", nargs="+", type=float, default=None)

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
        choices=["all", "only_text", "only_code", "random", "only_python"],
        default="",
    )
    parser.add_argument(
        "--neg_acts",
        nargs="+",
        type=str,
        choices=["all", "only_text", "only_code", "random", "only_python"],
        default="",
    )
    # this must have length of 2!, and x-axis mode must be first because
    # it determines the injection coefficients
    parser.add_argument(
        "--modes",
        nargs="+",
        type=str,
        choices=["all", "only_text", "only_code", "only_python"],
        default=["only_code", "only_text"],
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
        args.modes,
    )

    print(f"Eval object: {eval_obj.__dict__}")
    evaluate(eval_obj)


if __name__ == "__main__":
    main()
