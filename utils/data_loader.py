import tiktoken
import os

from torch.utils.data import Dataset


class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def load_data(self):
        data = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, "r") as f:
                    data.append(f.read())
        return data

    def preprocess_data(self, data):
        tokenized_data = []
        for text in data:
            token_count = len(self.tokenizer.encode(text))
            print(f"Token count: {token_count}")
            tokenized_data.append(self.tokenizer.encode(text))
        return tokenized_data


class TextDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.text) - self.sequence_length

    def __getitem__(self, index):
        return (
            self.text[index : index + self.sequence_length],
            self.text[index + 1 : index + self.sequence_length + 1],
        )
