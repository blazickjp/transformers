from idna import valid_contextj
import torch
from torch.utils.data import DataLoader
from utils.my_dataset import ShakespeareDataset
import numpy as np
from torch.nn import Transformer


# Check if a GPU is available and if so, set the device to GPU
device = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using {device} device")

# Define the hyperparameters for your model

emsize = 384  # Embedding dimension
nhid = (
    emsize * 4
)  # The dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6  # The number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 6  # The number of heads in the multiheadattention models
dropout = 0.2  # The dropout value
batch_size = 64
epochs = 1
seq_len = 256

with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

train_data = text[: int(0.9 * len(text))]
val_data = text[int(0.9 * len(text)) :]
train_dataset = ShakespeareDataset(train_data, seq_len)
val_dataset = ShakespeareDataset(val_data, seq_len)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


model = Transformer(
    d_model=emsize,
    nhead=nhead,
    dim_feedforward=nhid,
    num_encoder_layers=nlayers,
    num_decoder_layers=nlayers,
    dropout=dropout,
    norm_first=True,
    batch_first=True,
)
model = model.to("mps")
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


@torch.no_grad()
def estimate_loss():
    eval_iters = np.random.randint(175, 200)
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == "train":
                X, Y = next(iter(train_data_loader))
                print(X.shape, Y.shape)
            else:
                X, Y = next(iter(val_data_loader))
            logits = model(X, Y)
            loss = criterion(logits.view(-1, train_dataset.vocab_size), Y.view(-1))
            print(loss)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


loss = estimate_loss()
print(loss)

# Training loop

# for epoch in range(epochs):
#     model.train()  # Set the model to training mode
#     total_loss = 0
#     for iter, (input, target) in enumerate(train_data_loader):    # every once in a while evaluate the loss on train and val sets
# if iter % eval_interval == 0 or iter == max_iters - 1:
#     losses = estimate_loss()
#     print(
#         f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
#     )
#         if input.shape[0] != batch_size:
#             continue
#         # Move tensors to the correct device
#         input = input.to(device)
#         target = target.to(device)

#         # Forward pass
#         output = model(input)

#         # Calculate the batch loss
#         loss = criterion(output.view(-1, train_dataset.vocab_size), target.view(-1))

#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()

#         # Update weights
#         optimizer.step()

#         total_loss += loss.item()

#         if iter % 10 == 0:
#             print(f"Epoch: {epoch+1}, Loss: {total_loss/10}")
#             total_loss = 0
