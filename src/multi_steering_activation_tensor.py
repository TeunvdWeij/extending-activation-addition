import os
import torch


class MultiSteeringActivationTensor:
    """Class to save activations and its meta data"""

    def __init__(
        self,
        note,
        version,
        name,
        n_layers,
        device,
    ):
        self.note = note
        self.version = version
        self.name = name

        # create the placeholder for the activations based on the number of 
        # layers in model and the model's hidden dimension
        self.acts = torch.zeros((n_layers, 4096)).to(device)

    def __str__(self):
        return str(self.__dict__)

    def check_acts(self, acts):
        """Check whether values in acts are correct."""
        if torch.isinf(acts).any().item():
            raise ValueError("Infinite value in acts.")

        if torch.isnan(acts).any().item():
            raise ValueError("Nan value in acts.")


    def process_new_acts(self, new_acts, i, layer_idx):
        """
        new_acts: activations of model of certain layer
        i: sample index
        layer_idx: layer index
        """
        self.check_acts(new_acts)
        self.acts[layer_idx] = (self.acts[layer_idx] * i + new_acts) / (i + 1)

    def generate_save_path_string(self):
        folder = "data/activations/Llama-2-7b/multi_steering/"

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Path does not exist.\nThe folder path was: {folder}")
        save_path = folder + f"{self.name}.pt"

        if os.path.isfile(save_path):
            raise FileExistsError("File already exists, can't overwrite file.")

        print(f"SAVE PATH: {save_path}")
        self.save_path = save_path
        
    def save(self):
        self.generate_save_path_string()
        try:
            torch.save(self, self.save_path)
            print(f"\nSUCCESS: Tensor saved with metadata at {self.save_path}")
        except Exception as e:
            print(f"FAILED: Could not save the tensor. Error: {e}")
