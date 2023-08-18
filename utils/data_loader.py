import tiktoken
import os
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class DataLoader:
    """A class for loading and preprocessing text data.

    Attributes:
        data_dir (str): The directory where the text data is stored.
        tokenizer (Tokenizer): The tokenizer used to preprocess the text data.
    """

    def __init__(self, data_dir):
        """Initialize the DataLoader with the directory of the text data.

        Args:
            data_dir (str): The directory where the text data is stored.
        """
        self.data_dir = data_dir
        # self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def load_data(self):
        """Load text data from the specified directory.

        Returns:
            list: A list of strings, each representing the text from a single file.
        """
        data = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, "r") as f:
                    data.append(f.read())
        return data

    def preprocess_data(self, data):
        """Preprocess the loaded text data by tokenizing it.

        Args:
            data (list): A list of strings, each representing the text from a single file.

        Returns:
            list: A list of lists, each representing the tokenized text from a single file.
        """
        tokenized_data = []
        for text in data:
            token_count = len(self.tokenizer.encode(text))
            print(f"Token count: {token_count}")
            tokenized_data.append(self.tokenizer.encode(text))
        return tokenized_data


class TextDataset(Dataset):
    """A PyTorch Dataset for loading text data in a format suitable for training a transformer model.

    Attributes:
        text (list): The tokenized text data.
        sequence_length (int): The length of the sequences to be used for training.
    """

    def __init__(self, text, sequence_length):
        """Initialize the TextDataset with the text data and the sequence length for each training example.

        Args:
            text (list): The tokenized text data.
            sequence_length (int): The length of the sequences to be used for training.
        """
        assert (
            len(text) >= sequence_length
        ), "sequence_length must be less than or equal to the length of the text"
        self.text = text
        self.sequence_length = sequence_length

    def __len__(self):
        """Return the number of training examples in the dataset.

        Returns:
            int: The number of training examples in the dataset.
        """
        return len(self.text) - self.sequence_length

    def __getitem__(self, index):
        """Return the input and target sequence for the training example at the specified index.

        Args:
            index (int): The index of the training example.

        Returns:
            tuple: A tuple containing the input and target sequence for the training example at the specified index.
        """
        return (
            torch.tensor(self.text[index : index + self.sequence_length]),
            torch.tensor(self.text[index + 1 : index + self.sequence_length + 1]),
        )
