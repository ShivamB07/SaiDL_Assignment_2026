import torch
from torch.utils.data import Dataset, DataLoader

class WikiText2Dataset(Dataset):

    def __init__(self, tokenizer, block_size, split):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        from datasets import load_dataset

        text = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        encodings = tokenizer("\n\n".join(text["text"]), return_tensors="pt")
        encodings = encodings["input_ids"].squeeze()
        self.encodings = encodings

    def __len__(self):
        rem = len(self.encodings) % self.block_size
        return (len(self.encodings) - rem) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        input_ids = self.encodings[start:end]
        target_ids = self.encodings[start + 1 : end + 1]

        return input_ids, target_ids

def get_wikitext2_dataloader(tokenizer, block_size, split, batch_size, shuffle=True):
    dataset = WikiText2Dataset(tokenizer, block_size, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


