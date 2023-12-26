import json
import time
import os
import torch
import einops

from glob import glob
from torch.nn.functional import normalize

from model import Llama2Helper
from utils import get_hf_token, acc, get_model_name


class Evaluation:
    def __init__(
        self,
        note,
        version,
        total_tokens_per_ic,
        max_seq_length,
        chat,
        truncation,
        mean,
        ics,
        layers,
        model_params,
        dtype,
        pos_acts,
        neg_acts,
        modes,
    ):
        self.note = note
        self.version = version
        self.total_tokens_per_ic = total_tokens_per_ic
        self.max_seq_length = max_seq_length
        self.chat = chat
        self.truncation = truncation
        self.mean = mean

        if not self.mean and len(layers) > 1:
            raise ValueError("If mean is False, only one layer can be used")
        self.layers = layers
        self.model_params = model_params
        self.pos_acts = pos_acts
        self.neg_acts = neg_acts
        self.result_keys = [
            "top1_acc",
            "top10_acc",
            "skip50_top1_acc",
            "skip50_top10_acc",
            "total_encoded_tokens",
            "total_tokens_with_skip",
            "total_time_in_sec",
        ]

        self.save_path = self.generate_save_path_string()

        if dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError("Unimplemented dtype.")

        self.model_name = get_model_name(model_params, chat)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # set the range of injection coefficients to be tested.
        # 0 is prepended to get the model's performance without activation steering.
        if ics is not None:
            assert ics[0] == 0, "First injection coefficient must be 0!"
            self.ics = ics
            self.iter_over_all_ics = True
        else:
            self.ics = torch.cat(
                (
                    torch.tensor([0]),
                    torch.logspace(start=-1, end=3, base=10, steps=50, dtype=float),
                )
            ).to(self.device)
            self.iter_over_all_ics = False

        self.modes = modes

        self.acts = None
        self.results = {}

    def __str__(self) -> str:
        return str(self.__dict__)

    def get_model(self):
        model = Llama2Helper(
            model_name=self.model_name, hf_token=get_hf_token(), dtype=self.dtype
        )
        # TODO: check if this is needed
        self.model = model
        return model

    def set_used_ics(self, used_ics):
        self.used_ics = used_ics

    def get_meta_info(self):
        # Initialize result metadata.
        meta_data = {
            "model_name": self.model_name,
            "note": self.note,
            "total_tokens_per_ic": self.total_tokens_per_ic,
            "max_seq_length": self.max_seq_length,
            "layers": self.layers,
            "dtype": str(self.dtype),  # for json serialization
            "truncation": self.truncation,
            "mean": self.mean,
            "pos_acts": self.pos_acts,
            "neg_acts": self.neg_acts,
            "used_ics": self.used_ics,
            "modes": self.modes,
        }
        return meta_data

    def ic_results_template(self):
        # init dict for this injection coefficient
        ic_res = {
            "top1_acc": [],
            "top10_acc": [],
            "skip50_top1_acc": [],
            "skip50_top10_acc": [],
            "total_encoded_tokens": [],
            "total_skipped_tokens": [],
            "total_time_in_sec": [],
        }
        return ic_res

    def generate_save_path_string(self):
        save_path = f"results/coding/v_{self.version}.json"
        assert not os.path.isfile(
            save_path
        ), "File already exists, can't overwrite file."
        return save_path

    def encode(self, sample, mode):
        # truncate to context window, pad to longest sequence. detach and to device for gpu memory usage
        key = "content" if mode == "only_python" else "text"

        encoded = (
            self.model.tokenizer.encode(
                sample.get(key),
                truncation=self.truncation,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            .detach()
            .to(self.device)
        )
        return encoded

    def acts_compatibility_check(self, acts_list):
        if len(acts_list) == 0:
            print("No compatibility checked, acts_list is empty.")
            return

        to_check = [
            "num_samples",
            "max_seq_length",
            "truncation",
        ]
        if not self.mean:
            to_check.append("layers")

        for var in to_check:
            values = [getattr(act, var) for act in acts_list]
            if not values.count(values[0]) == len(values):
                raise RuntimeError(
                    "Activations do not match up. Maybe generated more activations with the same generation details."
                )

    def generate_random_acts(self):
        # TODO: remove this when 7.15 & 7.16 have been ran
        print("Warning, random activations have not been thoroughly tested.")
        if self.model_params == "7b":
            if self.mean:
                random_acts = torch.randn((32, 4096)).to(self.dtype).to(self.device)
                random_acts = normalize(random_acts, p=2, dim=1)
                return random_acts
            else:
                random_acts = torch.randn((1, 4096)).to(self.dtype).to(self.device)
                random_acts = normalize(random_acts, p=2, dim=1)
                return random_acts

        else:
            raise ValueError(("Random acts not implemented for this model."))

        assert self.acts is None, "Acts are not None, cannot overwwrite."

    def get_acts_path(self, mode):
        # finding the correct file name, bit hacky but is faster than loading many activations
        path = "data/activations/Llama-2-"
        path += self.model_params + "/"
        path += "mean/" if self.mean else "no-mean_/"
        path += f"mode={mode.replace('_', '-')}_"

        # path += f"layers-{'-'.join(str(item) for item in self.layers)}_"
        # print(f"The base path of the actication for {mode}: {path}")
        path_list = glob(path + "v*.pt")

        if len(path_list) == 0:
            raise RuntimeError(
                "No activations available. Change variables or generate new activations."
            )
        elif len(path_list) > 1:
            raise RuntimeError("Too many options, please clean activations.")
        else:
            return path_list[0]

    def get_acts_list(self, acts):
        """Takes in either self.pos_acts or self.neg_acts"""
        acts_list = []
        list_to_check = []

        for mode in acts:
            if mode == "random":
                acts_list.append(self.generate_random_acts())
            else:
                data = torch.load(self.get_acts_path(mode))
                list_to_check.append(data)
                acts_list.append(data.acts)

        acts_list = [a.to(self.device).to(self.dtype) for a in acts_list]
        return acts_list, list_to_check

    def set_acts(self):
        assert (
            self.pos_acts or self.neg_acts
        ), "No activations given, set a value for either pos_acts or neg_acts"
        assert (
            self.acts is None
        ), "Activations must be None, they cannot be overwritten."

        pos_acts, to_check_pos = self.get_acts_list(self.pos_acts)
        neg_acts, to_check_neg = self.get_acts_list(self.neg_acts)

        self.acts_compatibility_check(to_check_pos + to_check_neg)

        if pos_acts:
            pos_sum = einops.reduce(
                pos_acts, "mode layers model_dim -> layers model_dim", "sum"
            )
        else:
            pos_sum = torch.tensor((0))

        if neg_acts:
            neg_sum = einops.reduce(
                neg_acts, "mode layers model_dim -> layers model_dim", "sum"
            )
        else:
            neg_sum = torch.tensor((0))

        # set acts without normalizing
        self.acts = pos_sum - neg_sum

    def get_skip_tokens(self, mode="all", skip="skip50", data_type="tokens_int"):
        """Opens the skip tokens json file and returns a tensor"""
        with open("data/skip_tokens/skip_tokens.json", "r") as f:
            skip_tokens = json.load(f)

        return torch.tensor(skip_tokens[mode][skip][data_type]).to(self.device)

    def process_predictions(self, encoded, predictions, start_time, skip_tokens):
        # align predictions: the first token is not predicted by the model
        # and the last prediction is not encoded
        encoded = encoded[:, 1:]
        predictions = predictions[:, :-1]

        top1_preds = torch.topk(predictions, k=1, dim=-1).indices.to(self.device)
        top10_preds = torch.topk(predictions, k=10, dim=-1).indices.to(self.device)
        top1_acc = acc(encoded, top1_preds)
        top10_acc = acc(encoded, top10_preds, top1=False)

        # create filter checking for tokens being in self.skip_tokens

        f_50 = ~(encoded.unsqueeze(-1) == skip_tokens.any(-1)).squeeze(-1)
        total_tokens_with_skip = torch.sum(f_50).item()

        # after skipping it could leave sample with 0 tokens.
        # in this case, set acc to 0 for easier handling.
        # note that the *weighted* acc is later calculated, so
        # this has no effect on the overall accuracy
        if total_tokens_with_skip > 0:
            skip50_top1_acc = acc(encoded, top1_preds, f_50)
            skip50_top10_acc = acc(encoded, top10_preds, f_50, top1=False)
        else:
            skip50_top1_acc, skip50_top10_acc = 0, 0

        return (
            top1_acc,
            top10_acc,
            skip50_top1_acc,
            skip50_top10_acc,
            encoded.numel(),
            total_tokens_with_skip,
            round(time.perf_counter() - start_time, 3),
        )

    def save(self):
        self.results["meta"] = self.get_meta_info()
        with open(self.save_path, "w") as f:
            json.dump(self.results, f, indent=2)
            print("Written to json file succesfully!")
