import torch


class ActivationTensor:
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

        self.contains_inf = True if torch.isinf(self.tensor).any() else False
        self.contains_nan = True if torch.isnan(self.tensor).any() else False

    def save(self):
        torch.save(self, self.file_path)
        print("SUCCES: Tensor saved with metadata")
