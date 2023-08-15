import pytest
import tiktoken
from utils.data_loader import DataLoader


class TestDataLoader:
    @pytest.fixture
    def data_loader(self):
        data_dir = "./tests/test_data"  # directory with test data
        return DataLoader(data_dir)

    def test_load_data(self, data_loader):
        data = data_loader.load_data()
        assert isinstance(data, list)
        assert all(isinstance(item, str) for item in data)

    def test_preprocess_data(self, data_loader):
        data = ["This is a test sentence.", "This is another test sentence."]
        tokenized_data = data_loader.preprocess_data(data)
        print(f"Tokenized Data: {tokenized_data}")
        assert isinstance(tokenized_data, list)
        # Check that the values in the dictionary are PyTorch tensors
        for item in tokenized_data:
            print(f"item: {item}")
            assert isinstance(item, list)

        # # Check that the tokenized data can be decoded back into the original text
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        for item in tokenized_data:
            decoded_text = tokenizer.decode(item)
            print(decoded_text)
            assert decoded_text in data
