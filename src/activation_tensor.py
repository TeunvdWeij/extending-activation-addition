import torch


class ActivationTensor:
    """Class to save activations and its meta data"""
    def __init__(
        self,
        tensor: torch.Tensor,
        num_samples: int,
        file_path: str,
        mode: str,
        model_name: str,
        layer: int,
        max_seq_length: int,
        truncation: bool,
        total_time: float,
        total_tokens: int,
        note: str,
    ):
        self.tensor = tensor
        self.num_samples = num_samples
        self.file_path = file_path
        self.mode = mode
        self.model_name = model_name
        self.layer = layer
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.total_time = total_time
        self.total_tokens = total_tokens
        self.note = note

        self.contains_inf = self.check_tensor(torch.isinf)
        self.contains_nan = self.check_tensor(torch.isnan)

    
    def check_tensor(self, f):
        """Checks the tensor using the passed function and return the status."""
        return f(self.tensor).any().item()

    def save(self):
        """Saves the tensor to file and prints success message or handles exception."""
        try:
            torch.save(self, self.file_path)
            print(f"SUCCESS: Tensor saved with metadata at {self.file_path}")
        except Exception as e:
            print(f"FAILED: Could not save the tensor. Error: {e}")
