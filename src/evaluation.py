import torch
import json
import time
import os

from model import Llama2Helper
from utils import (
    load_pile,
    get_hf_token,
    get_skip_tokens,
    acc,
)

# TODO: probs work with parser
class Evaluation:
    def __init__(self, model: Llama2Helper):
        self.total_tokens_per_ic = 100_000
        self.layer = 29
        self.max_seq_length = 4096
        self.injection_coefficients = (0, 20, 40, 75, 100, 150, 200, 250, 300, 400, 500)
        self.modes = ("only_text", "only_code")
        self.version = "3.00" #TODO: this is probs not how it should work

        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.results = {}
        self.acts = None

        # self.pos_act_file_path = "data/activations/Llama-2-7b-chat-hf_v2.01.pt"
        # self.neg_act_file_path = "data/activations/Llama-2-7b-chat-hf_v2.02.pt"

        self.hf_token = get_hf_token()
        self.init_results_meta()

    def init_results_meta(self):
        # Initialize result metadata.
        self.results["meta"] = {
            "model_name": self.model_name,
            "layer": self.layer,
            "max_seq_length": self.max_seq_length,
            "injection_coefficients": self.injection_coefficients,
            "total_tokens_per_ic": self.total_tokens_per_ic,
            "note": "test with new acts, although they contain infs",
        }
    
    def generate_results_ic(self):
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
        
    def set_total_tokens_per_ic(self, value:int):
        if isinstance(value, int) and value >= 0:
            self.total_tokens_per_ic = value
        else:
            raise ValueError("The total_tokens_per_ic should be a nonnegative integer.")

    def set_layer(self, value:int):
        if isinstance(value, int) and value >= 0:
            self.layer = value
        else:
            raise ValueError("The layer should be a nonnegative integer.")

    def set_max_seq_length(self, value:int):
        if isinstance(value, int) and value >= 0:
            self.max_seq_length = value
        else:
            raise ValueError("The max_seq_length should be a nonnegative integer.")

    def set_injection_coefficients(self, values:tuple):
        if isinstance(values, tuple) and all(isinstance(v, (int, float)) and v >= 0 for v in values):
            self.injection_coefficients = values
        else:
            raise ValueError("The injection_coefficients should be a tuple of nonnegative integers or floats.")

    def create_save_path(self):
        self.save_path = f"data/activations/{model_name.split('/')[1]}_{mode.replace('_', '-')}_v2.08.pt"

    def check_if_file_exists(self):
        self.create_save_path()
        assert not os.path.isfile(self.save_path), "File already exists, rerun experiment with different version."

    def set_random_activations(self):
        assert self.acts is None, "Acts are not None, cannot overwwrite."
        
        # some dummy input to get the shape of layer
        self.model.get_logits(torch.tensor([[1]]))

        acts_shape = self.model.get_last_activations(layer).shape
        random_acts = torch.rand(acts_shape).to(model.dtype).to(model.device)
        self.acts = random_acts / torch.norm(random_acts, p=2)

    def set_acts(self, acts):
        assert self.acts is None, "Acts are not None, cannot overwwrite."
        self.acts = acts
 
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

        with open("data/skip_tokens.json", "r") as f:
            skip_tokens = json.load(f)

        return torch.tensor(skip_tokens[mode][skip][data_type]).to()

    def 
    
    def main_loop(self):
        for mode in self.modes:
            self.results[mode] = {}
            skip_tokens = self.get_skip_tokens(mode=mode, skip="skip50", data_type="tokens_int")
            dataset = load_pile(split="validation", mode=mode, iterable=True)

            for ic in self.injection_coefficients:
                print(f"Mode: {mode}.   Injection Coefficient: {ic}", flush=True)
                # clear and set the activations to be used in the forward pass
                self.model.reset_all()
                self.model.set_add_activations(layer, ic*(self.acts))
                ic_res = self.generate_results_ic()

                analyzed_tokens = 0
                for sample in self.dataset:
                    if analyzed_tokens > total_tokens_per_ic:
                        break

                    start_time = time.perf_counter()

                    encoded = (
                        self.model.tokenizer.encode(
                            sample.get("text"),  # type: ignore
                            truncation=True,
                            max_length=max_seq_length,
                            return_tensors="pt",
                        )
                        .detach() # type: ignore
                        .to(device)
                        )
                    predictions = self.model.get_logits(encoded).detach().to(device)

                    # align predictions: the first token is not predicted by the model
                    # and the last prediction is not encoded
                    encoded = encoded[:, 1:]
                    predictions = predictions[:, :-1]

                    top1_preds = torch.topk(predictions, k=1, dim=-1).indices.to(device)
                    top10_preds = torch.topk(predictions, k=10, dim=-1).indices.to(device)
                    top1_acc = acc(encoded, top1_preds)
                    top10_acc = acc(encoded, top10_preds, top1=False)

                    # create filter which also checks whether true tokens are in skip50
                    f_50 = ~(encoded.unsqueeze(-1) == skip_tokens).any(-1)

                    skip50_top1_acc = acc(encoded, top1_preds, f_50)
                    skip50_top10_acc = acc(encoded, top10_preds, f_50, top1=False)
            
                    ic_res["top1_acc"].append(top1_acc)
                    ic_res["top10_acc"].append(top10_acc)
                    ic_res["skip50_top1_acc"].append(skip50_top1_acc)
                    ic_res["skip50_top10_acc"].append(skip50_top10_acc)
                    ic_res["total_encoded_tokens"].append(encoded.numel())
                    ic_res["total_skipped_tokens"].append(torch.sum(f_50).item())
                    ic_res["total_time_in_sec"].append(
                        round(time.perf_counter() - start_time, 3)
                    )

                    analyzed_tokens += encoded.numel()

                results[mode][f"injection_coefficients_{ic}"] = ic_res


with open(file_path, "w") as f:
    json.dump(results, f, indent=2)
    print(f"Written to json file succesfully!")
