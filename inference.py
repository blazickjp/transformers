import torch
from models.transformer import TransformerModel
from utils.data_loader import DataLoader as CustomDataLoader

# Load the trained model
ntokens = 100253  # The size of vocabulary
emsize = 768  # Embedding dimension
nhead = 12  # The number of heads in the multiheadattention models
nhid = 100  # The dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # The number of nn.TransformerEncoderLayer in nn.TransformerEncoder
dropout = 0.2  # The dropout value

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
model.load_state_dict(torch.load("transformer_model.pth"))
model.eval()

# Load and preprocess the data
custom_data_loader = CustomDataLoader()
preprocessed_data = custom_data_loader.preprocess_data(["MARCIUS:"])

# Convert the list of tokenized data to a single list
text = [token for sublist in preprocessed_data for token in sublist]

# Convert the text to tensor
text_tensor = torch.tensor(text).unsqueeze(0)  # Add batch dimension

# Generate predictions
with torch.no_grad():
    output = model(text_tensor)
    predictions = torch.argmax(
        output, dim=2
    )  # Get the index of the max log-probability

print(predictions)
