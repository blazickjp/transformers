import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import TextDataset
from models.transformer import TransformerModel

# Define the hyperparameters for your model
ntokens = 128  # The size of vocabulary
emsize = 200  # Embedding dimension
nhid = 200  # The dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # The number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # The number of heads in the multiheadattention models
dropout = 0.2  # The dropout value
batch_size = 64
epochs = 10

# Initialize the model
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
if torch.cuda.is_available():
    model.cuda()
    print("Using GPU")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Load and preprocess the data
dataset = TextDataset("data/shakespeare.txt", 30)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    for i, (input, target) in enumerate(data_loader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # Forward pass
        output = model(input)
        loss = criterion(output.view(-1, ntokens), target.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch: {epoch}, Loss: {total_loss / len(data_loader)}")

# Save the trained model
torch.save(model.state_dict(), "transformer_model.pth")
