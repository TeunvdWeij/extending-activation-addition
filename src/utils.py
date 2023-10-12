from datasets import load_dataset


# adapted from https://github.com/pesvut/separability/blob/b435310c5728dcfacb0312799d98ba6e7146507c/src/separability/texts.py#L3
def load_pile(split, mode):
    """
    Load pile given certain selection.

    :param split: The split of dataset to load
    :param mode: The mode which can be one of "all", "only_text", "only_code"

    :return dataset: The requested dataset based on the selection
    """

    assert mode in ("all", "only_text", "only_code")
    dataset = load_dataset("monology/pile-uncopyrighted", streaming=True, split=split)

    if mode == "all":
        return dataset
    if mode == "only_text":
        return dataset.filter(
            lambda sample: sample["meta"]["pile_set_name"] != "Github"
        )
    if mode == "only_code":
        return dataset.filter(
            lambda sample: sample["meta"]["pile_set_name"] == "Github"
        )


def get_subset_from_dataset(dataset, num_samples):
    """
    Get a specific number of samples from the  dataset.

    :param dataset: The dataset to sample from
    :param num_samples: The number of samples to extract

    :return sampled_text: A list of 'num_samples' text strings from the dataset
    """
    sampled_text = []
    for i, batch in enumerate(dataset.shuffle(seed=13, buffer_size=num_samples)):
        if i > num_samples:
            break
        sampled_text.append(batch["text"])
    return sampled_text
