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
        layers,
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
        self.layers = layers
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.mean = mean

        if dtype == "float16":
            self.dtype = torch.float16
        if dtype == "bfloat16":
            self.dtype = torch.bfloat16

        self.model_name = get_model_name(model_params, chat)

        # either store the mean or save acts in list
        self.acts = torch.tensor(0) if mean else []
        self.save_path = self.generate_save_path_string()

    def __str__(self):
        return self.__dict__
        
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
        mean_name = "mean" if self.mean else "no_mean"
        save_path = (
            f"{folders}{self.mode.replace('_', '-')}_{mean_name}_v{self.version}.pt"
        )
        print(folders)
        assert os.path.isdir(folders), "Path does not exists."
        assert not os.path.isfile(
            save_path
        ), "File already exists, can't overwrite file."
        return save_path

    def process_new_acts(self, new_acts, i):
        self.check_acts(new_acts)

        if self.mean:
            self.acts = (self.acts * i + new_acts) / (i + 1)
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
