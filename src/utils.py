from datasets import load_dataset


# adapted from https://github.com/pesvut/separability/blob/b435310c5728dcfacb0312799d98ba6e7146507c/src/separability/texts.py#L3
def load_pile(split, only_code=False):
    dataset = load_dataset("monology/pile-uncopyrighted", streaming=True, split=split)

    if not only_code:
        return dataset

    else:

        def filter_out_code(example):
            return example["meta"]["pile_set_name"] != "Github"

        dataset = dataset.filter(filter_out_code)
        return dataset


def get_subset_from_dataset(dataset, num_samples):
    much_text = []
    for i, batch in enumerate(dataset.shuffle(seed=13, buffer_size=num_samples)):
        if i > num_samples:
            break
        much_text.append(batch["text"])
    return much_text
