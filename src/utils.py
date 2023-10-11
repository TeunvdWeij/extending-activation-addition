from datasets import load_dataset

# adapted from https://github.com/pesvut/separability/blob/b435310c5728dcfacb0312799d98ba6e7146507c/src/separability/texts.py#L3  
def load_pile(split, include_code=True):
    dataset = load_dataset("monology/pile-uncopyrighted", streaming=True, split=split)
    
    if include_code:
        return dataset

    else:
        def filter_out_code(example):
            return example['meta']['pile_set_name'] != 'Github'
        dataset = dataset.filter(filter_out_code)
        return dataset
