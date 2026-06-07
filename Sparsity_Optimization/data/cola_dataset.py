import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
 
 
# FIX 6: Class was named `Dataset`, which:
#   (a) shadows the imported `Dataset` from torch.utils.data, and
#   (b) doesn't match the name used in get_cola_dataloader → `COLADataset(...)` → NameError
class COLADataset(Dataset):
    def __init__(self, tokenizer, block_size, split):
        self.tokenizer = tokenizer
        self.block_size = block_size
 
        dataset = load_dataset("nyu-mll/glue", "cola", split=split)
        self.encodings = []
        self.attention_masks = []
        self.labels = []
        for item in dataset:
            tokenized = tokenizer(
                item["sentence"],
                truncation=True,
                padding="max_length",
                max_length=block_size,
            )
            self.encodings.append(tokenized["input_ids"])
            self.attention_masks.append(tokenized["attention_mask"])
            self.labels.append(item["label"])
 
    def __len__(self):
        return len(self.encodings)
 
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
 
 
def get_cola_dataloader(tokenizer, split, batch_size, block_size=128, shuffle=True):
    dataset = COLADataset(tokenizer, block_size, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)