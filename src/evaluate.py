import torch
import time
import argparse

from evaluation import Evaluation

from utils import load_pile


def evaluate(eval_obj: Evaluation):
    model = eval_obj.get_model()
    eval_obj.set_random_acts()

    for mode in eval_obj.modes:
        eval_obj.results[mode] = {}
        skip_tokens = eval_obj.get_skip_tokens(mode)
        dataset = load_pile(split="validation", mode=mode, iterable=True)

        for ic in eval_obj.ics:
            print(f"Mode: {mode}.   Injection Coefficient: {ic}", flush=True)
            # clear and set the activations to be used in the forward pass
            model.reset_all()
            for layer in eval_obj.layers:
                model.set_add_activations(layer, ic * eval_obj.acts)

            # init dict for this injection coefficient
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
                    encoded, predictions, start_time, skip_tokens,
                )
                ic_results.append(sample_results)

                analyzed_tokens += encoded.numel()

            eval_obj.results[mode][f"injection_coefficients_{ic}"] = ic_results

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
        "--pos_acts", nargs="+", type=str, choices=["all", "only_text", "only_code", "random"], default=""
    )    
    parser.add_argument(
        "--neg_acts", nargs="+", type=str, choices=["all", "only_text", "only_code", "random"], default=""
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


# ARCHIVE
pos_act_file_path = "data/activations/Llama-2-7b/Llama-2-7b-chat-hf_only-text_2.04.pt"
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
# random_acts = torch.rand(acts_shape).to(model.dtype).to(model.device)
# acts = random_acts / torch.norm(random_acts, p=2)


# acts_only_code = torch.load("data/activations/Llama-2-7b/only-code_no_mean_v2.17.pt", map_location="cpu").acts
# acts_only_code = normalize(acts_only_code, p=2, dim=1)
# pca = PCA(n_components=1)
# pca_only_code = pca.fit(acts_only_code.float().numpy()).components_

# acts_only_text = torch.load("data/activations/Llama-2-7b/only-text_no_mean_v2.18.pt", map_location="cpu").acts
# acts_only_text = normalize(acts_only_text, p=2, dim=1)
# pca = PCA(n_components=1)
# pca_only_text = pca.fit(acts_only_text.float().numpy()).components_

# acts_all = torch.load("data/activations/Llama-2-7b/all_no_mean_v2.16.pt", map_location="cpu").acts
# acts_all = normalize(acts_all, p=2, dim=1)
# pca = PCA(n_components=1)
# pca_all = pca.fit(acts_all.float().numpy()).components_

# # acts = pca_only_text - pca_only_code - pca_all
# acts = pca_only_text - pca_only_code
# acts = pca_only_text - pca_only_code - pca_all
# acts = torch.tensor(-pca_only_code + pca_only_text - pca_all).to(model.dtype).to(device)
# print(f"Acts shape: {acts.shape}")
