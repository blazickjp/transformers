import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from utils.data_loader import DataLoader as CustomDataLoader, TextDataset
from models.transformer import TransformerModel
from tqdm import tqdm


# Check if a GPU is available and if so, set the device to GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define the hyperparameters for your model
emsize = 768  # Embedding dimension
nhid = 100  # The dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # The number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 12  # The number of heads in the multiheadattention models
dropout = 0.2  # The dropout value
batch_size = 64
epochs = 1

# Load and preprocess the data
custom_data_loader = CustomDataLoader("data")

# Load the data from the file
data = custom_data_loader.load_data()

# Preprocess the data
preprocessed_data = custom_data_loader.preprocess_data(data)

# Convert the list of tokenized data to a single list
text = [token for sublist in preprocessed_data for token in sublist]
ntokens = max(text) + 1  # The size of vocabulary

# Initialize the model
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


# Print the maximum token index
print(f"Max token index: {max(text)}")

# Pass the text to the TextDataset
dataset = TextDataset(text, 20)
data_loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    pbar = tqdm(total=len(data_loader), desc=f"Training Epoch: {epoch+1}", unit="batch")
    for i, (input, target) in enumerate(data_loader):
        # Move tensors to the correct device
        input = input.to(device)
        target = target.to(device)

        # Reshape the input and target data to be 3-D tensors
        input = input.view(batch_size, -1)
        # Forward pass
        output = model(input)
        # target = target.view(batch_size)
        output_reshaped = output.view(-1, output.shape[-1])
        target_reshaped = target.view(-1)
        loss = criterion(output_reshaped, target_reshaped)

        # Calculate the loss
        loss = criterion(output_reshaped, target_reshaped)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({"Loss": total_loss / (i + 1)})

    print(f"Epoch: {epoch}, Loss: {total_loss / len(data_loader)}")

# Save the trained model
torch.save(model.state_dict(), "transformer_model.pth")
