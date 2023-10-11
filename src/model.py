# file containing code related to model
# adapted from https://github.com/nrimsky/LM-exp/blob/main/sycophancy/sycophancy_steering.ipynb. 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

with open("private_information/hf_token", "r") as f:
    hf_token = f.read()


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.last_hidden_state = None
        self.add_activations = None
        self.output_init = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.last_hidden_state = output[0]
        self.output_before_adding = output
        if self.add_activations is not None:
            output = (output[0] + self.add_activations,) + output[1:]
        self.output_after_adding = output
        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.last_hidden_state = None
        self.add_activations = None


class Llama27BHelper:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = init_tokenizer(model_name)
        # not sure what this is used for tbh
        self.tokenizer.pad_token = self.tokenizer.eos_token


        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            token=hf_token,
            torch_dtype=torch.half,
        )

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer)

    def generate_text(self, prompt, do_sample=False, temperature=1.0, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(
            inputs.input_ids.to(self.device),
            do_sample=do_sample,
            temperature=temperature,
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def get_logits(self, tokens):
        with torch.no_grad():
            return self.model(tokens.to(self.device)).logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].last_hidden_state

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

def init_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(
                model_name,
                device_map="auto",
                token=hf_token,
                torch_dtype=torch.half,
            )