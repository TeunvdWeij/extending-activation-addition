import os
import torch

from utils import get_model_name


class ActivationTensor:
    """Class to save activations and its meta data"""

    def __init__(
        self,
        note,
        version,
        num_samples,
        mode,
        model_params,
        chat,
        layer_idx,
        max_seq_length,
        truncation,
        mean,
        dtype,
    ):
        self.note = note
        self.version = version
        self.num_samples = num_samples
        self.mode = mode
        self.model_params = model_params
        self.chat = chat
        self.layer_idx = layer_idx
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.mean = mean

        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.model_name = get_model_name(model_params, chat)
        self.save_path = self.generate_save_path_string()

    def __str__(self):
        return str(self.__dict__)

    def set_acts(self, layers: int = None):
        # either store the mean or save acts in list
        if not self.mean:
            self.acts = []
        else:
            # store the activations for each layer, hidden dim is 4096
            self.acts = torch.zeros((layers, 4096))

    def check_acts(self, acts):
        """Check whether values in acts are correct."""
        if torch.isinf(acts).any().item():
            raise ValueError("Infinite value in acts.")

        if torch.isnan(acts).any().item():
            raise ValueError("Nan value in acts.")

    def set_total_time(self, total_time):
        self.total_time = total_time

    def set_total_tokens(self, total_tokens):
        self.total_tokens = total_tokens

    def generate_save_path_string(self):
        folders = f"data/activations/Llama-2-{self.model_params}/"
        folders += "mean/" if self.mean else "no_mean/"

        if not self.mean:
            mean_str = f"layer={self.layer_idx}"
            save_path = f"{folders}mode={self.mode.replace('_', '-')}_{mean_str}_v{self.version}.pt"
        else:
            save_path = (
                f"{folders}mode={self.mode.replace('_', '-')}_v{self.version}.pt"
            )
        if not os.path.isdir(folders):
            raise FileNotFoundError("Path does not exist.")
        if os.path.isfile(save_path):
            raise FileExistsError("File already exists, can't overwrite file.")

        return save_path

    def process_new_acts(self, new_acts, i, layer_idx=None):
        """
        new_acts: activations of model of certain layer
        i: sample index
        layer_idx: layer index, only needed when mean is True
        """
        self.check_acts(new_acts)

        if self.mean:
            self.acts[layer_idx] = (self.acts[layer_idx] * i + new_acts) / (i + 1)
        else:
            self.acts.append(new_acts)

    def save(self):
        if not self.mean:
            self.acts = torch.vstack(self.acts)
        try:
            torch.save(self, self.save_path)
            print(f"SUCCESS: Tensor saved with metadata at {self.save_path}")
        except Exception as e:
            print(f"FAILED: Could not save the tensor. Error: {e}")
