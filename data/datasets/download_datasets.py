# code adapted from https://github.com/nrimsky/ActivationDirectionAnalysis/blob/main/datasets/download_datasets.py

import requests
import json
import random
from random import sample

random.seed(13)

urls = {
    "sycophancy": "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/sycophancy_on_political_typology_quiz.jsonl",
    "wealth_seeking": "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/advanced-ai-risk/human_generated_evals/wealth-seeking-inclination.jsonl",
    "myopic": "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/advanced-ai-risk/human_generated_evals/myopic-reward.jsonl",
    "agreeableness": "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/persona/agreeableness.jsonl",
    "anti_immigration": "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/persona/anti-immigration.jsonl",
}
N_SAMPLES = 1000


def get_data(n_samples, urls):
    for name, url in urls.items():
        r = requests.get(url).text
        raw_data = [json.loads(line) for line in r.split("\n") if line != ""]
        data = []
        for d in raw_data:
            am = d["answer_matching_behavior"].strip().replace("(", "").replace(")", "")
            anm = (
                d["answer_not_matching_behavior"]
                .strip()
                .replace("(", "")
                .replace(")", "")
            )
            question = (
                d["question"].replace("Question:", "").replace("Answer:", "").strip()
            )
            # these are yes-no questions
            if name in ("agreeableness", "anti_immigration"):
                question = (
                    f"Human:\n{question}\n\nAI:\nOut of (Yes) and (No), I choose ("
                )
            else:
                question = f"Human:\n{question}\n\nAI:\nOut of (A) and (B), I choose ("
            data.append(
                {
                    "answer_matching_behavior": am,
                    "answer_not_matching_behavior": anm,
                    "question": question,
                }
            )
        if len(data) > n_samples:
            data = sample(data, n_samples)
        with open(f"data/datasets/{name}.json", "w") as f:
            json.dump(data, f)


get_data(N_SAMPLES, urls)
