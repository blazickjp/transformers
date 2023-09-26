import torch
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
device = "mps"


class ShakespeareDataset(Dataset):
    def __init__(self, text, sequence_length):
        super().__init__()
        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        self.sequence_length = sequence_length
        self.vocab_size = len(tokenizer)

    def __len__(self):
        return len(self.tokens) - (self.sequence_length + 1)

    def __getitem__(self, index):
        start_idx = index
        end_idx = start_idx + self.sequence_length
        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx + 1 : end_idx + 1]
        return torch.tensor(x).to(device), torch.tensor(y).to(device)


# with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# dataset = ShakespeareDataset(text, 256)
# data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
# x, y = next(iter(data_loader))
