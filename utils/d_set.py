from dataclasses import dataclass
import torch
from utils.config import Config


@dataclass
class DSet:
    text: str
    chars: list
    stoi: dict
    itos: dict
    data: torch.Tensor
    train_data: torch.Tensor
    val_data: torch.Tensor

    def __init__(self, filepath, config: Config, split_ratio=0.9):
        with open(filepath, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.data = torch.tensor([self.stoi[c] for c in self.text], dtype=torch.long)

        n = int(split_ratio * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        self.vocab_size = len(self.chars)
        self.config = config

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(
            len(data) - self.config.block_size, (self.config.batch_size,)
        )
        x = torch.stack([data[i : i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.config.block_size + 1] for i in ix])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y
