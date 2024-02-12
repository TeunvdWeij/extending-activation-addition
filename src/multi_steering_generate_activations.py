import argparse

from model import Llama2Helper
from utils import get_hf_token, encode, load_multi_steering_data, get_model_name
from multi_steering_activation_tensor import MultiSteeringActivationTensor


def get_activations(note, version, params, chat):
    model_name = get_model_name(params, chat)
    model = Llama2Helper(
        model_name=model_name,
        hf_token=get_hf_token(),
    )
    datasets = load_multi_steering_data()

    for name, data in datasets.items():
        acts = MultiSteeringActivationTensor(
            note, version, name, model.n_layers, model.device
        )
        # :500 to take as training data for the steering vectors, note that they are randomized before
        for sample_idx, row in enumerate(data[:500]):
            q = row["question"]
            match_q = q + row["answer_matching_behavior"]
            encoded = encode(model, match_q)

            # get the activations associated to dataset behaviour
            model.get_logits(encoded)
            matching_acts = []
            for layer_idx in range(model.n_layers):
                activations = model.get_last_activations(layer_idx)
                activations = activations[0, -1, :].detach()
                matching_acts.append(activations)

            # get the activations associated to non-sycopahntic behaviour
            non_match_q = q + row["answer_not_matching_behavior"]
            encoded = encode(model, non_match_q)

            model.get_logits(encoded)
            non_matching_acts = []
            for layer_idx in range(model.n_layers):
                activations = model.get_last_activations(layer_idx)
                activations = activations[0, -1, :].detach()
                non_matching_acts.append(activations)

            # get the difference in activations and calculate mean
            for layer_idx in range(model.n_layers):
                new_acts = (matching_acts[layer_idx] - non_matching_acts[layer_idx]).to(
                    model.device
                )
                acts.process_new_acts(new_acts, sample_idx, layer_idx)

        acts.save()


def arg_parser():
    parser = argparse.ArgumentParser()

    # required params
    parser.add_argument("--note", type=str)
    parser.add_argument("--version", type=str)

    # optional params
    parser.add_argument("--chat", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--model_params", type=str, choices=["7b", "13b", "70b"], default="7b"
    )

    return parser.parse_args()


def main():
    args = arg_parser()
    get_activations(
        args.note,
        args.version,
        args.model_params,
        args.chat,
    )


if __name__ == "__main__":
    main()
