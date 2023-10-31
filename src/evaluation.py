import torch
import json
import time
import os

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
    ):
        self.note = note
        self.version = (version,)
        self.total_tokens_per_ic = total_tokens_per_ic
        self.max_seq_length = max_seq_length
        self.chat = chat
        self.truncation = truncation
        self.mean = mean
        self.ics = ics
        self.layers = layers
        self.model_params = model_params
        self.pos_acts = pos_acts
        self.neg_acts = neg_acts

        if dtype == "float16":
            self.dtype = torch.float16
        if dtype == "bfloat16":
            self.dtype = torch.bfloat16

        self.model_name = get_model_name(model_params, chat)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.modes = ("only_text", "only_code")

        self.acts = None
        self.results = {}
        self.results["meta"] = self.get_meta_info()

    def __str__(self) -> str:
        return str(self.__dict__)
    
    def get_model(self):
        model = Llama2Helper(
            model_name=self.model_name, hf_token=get_hf_token(), dtype=self.dtype
        )
        # TODO: check if this is needed
        self.model = model
        return model

    def get_meta_info(self):
        # Initialize result metadata.
        meta_data = {
            "model_name": self.model_name,
            "note": self.note,
            "total_tokens_per_ic": self.total_tokens_per_ic,
            "max_seq_length": self.max_seq_length,
            "layers": self.layers,
            "injection_coefficients": self.ics,
            "dtype": self.dtype,
            "truncation": self.truncation,
            # TODO: this should be different
            "mean": self.mean,
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
        save_path = f"results/Llama-2-{self.model_params}/v_{self.version}.json"
        assert os.path.isdir(save_path), "Directory does not exists."
        assert not os.path.isfile(
            save_path
        ), "File already exists, can't overwrite file."
        return save_path

    def encode(self, sample):
        # truncate to context window, pad to longest sequence. detach and to device for gpu memory usage
        encoded = (
            self.model.tokenizer.encode(
                sample.get("text"),  # type: ignore
                truncation=self.truncation,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            .detach()  # type: ignore
            .to(self.device)
        )
        return encoded
    
    def acts_compatibility_check(acts):
        #TODO: should change to layers with new activation generation
        to_check = (
            "num_samples", 
            "layer", 
            "max_seq_length",
            "truncation",
        )

        for var in to_check:
            values = [getattr(act, var) for act in acts]
            print(values)
            if not values.count(values[0]) == len(values):
                raise RuntimeError("Activations do not match up. Maybe generated more activations with the same generation details.")


    def generate_random_acts(self):
        assert self.acts is None, "Acts are not None, cannot overwwrite."

        # some dummy input to get the shape of layer
        self.model.get_logits(torch.tensor([[1]]))

        # all layers have the same shape
        acts_shape = self.model.get_last_activations(self.layers[0]).shape
        random_acts = torch.rand(acts_shape).to(self.dtype).to(self.device)
        random_acts = normalize(random_acts, p=2, dim =1)
        print(random_acts.shape, print(random_acts))
        return random_acts

    def get_acts_path(self, mode):
        # finding the correct file name, bit hacky but is faster than loading many activations
        path = "data/activations/Llama-2"
        path += self.model_params
        path += mode.replace("_", "-")
        if self.mean:
            path += "_mean_"
        else:
            path += "_no-mean_"

        path_list = glob(path + "v*.pt")
        if len(path_list) == 0:
            raise RuntimeError("No activations available. Change variables or generate new activations.")
        elif len(path_list) > 1:
            raise RuntimeError("Too many options, please clean activations.")
        else:
            return path_list[0]
    
    def set_acts(self):
        assert self.pos_acts or self.neg_acts, "No activations given, set a value for either pos_acts or neg_acts"
        assert self.acts is None, "Activations must be None, they cannot be overwritten."

        pos_acts, neg_acts = [], []
        list_to_check = []
        for mode in self.pos_acts:
            if mode == "random":
                pos_acts.append(self.set_random_acts())
            else:
                data = torch.load(self.get_acts_path(mode))
                list_to_check.append(data)
                pos_acts.append(data.acts)

        for mode in self.neg_acts:
            if mode == "random":
                neg_acts.append(self.set_random_acts())
            else:
                data = torch.load(self.get_acts_path(mode))
                list_to_check.append(data)
                neg_acts.append(data.acts)

        self.acts_compatibility_check(list_to_check)

        if pos_acts:
            acts = torch.sum(torch.vstack(tuple(pos_acts)), dim=0)
            if neg_acts:
                acts -= torch.sum(torch.vstack(tuple(neg_acts)), dim=0)
        else:
            acts = -torch.sum(torch.vstack(tuple(neg_acts)), dim=0)

        self.acts = acts

    def get_skip_tokens(self, mode="all", skip="skip50", data_type="tokens_int"):
        """Opens the skip tokens json file and returns a tensor"""
        with open("data/skip_tokens.json", "r") as f:
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
        with open(self.file_path, "w") as f:
            json.dump(self.results, f, indent=2)
            print(f"Written to json file succesfully!")
