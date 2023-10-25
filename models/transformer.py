import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import Config


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, config: Config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.n_embd // config.n_head, bias=False)
        self.query = nn.Linear(
            config.n_embd, config.n_embd // config.n_head, bias=False
        )
        self.value = nn.Linear(
            config.n_embd, config.n_embd // config.n_head, bias=False
        )
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out, wei


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, config: Config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(
            (config.n_embd // config.n_head) * config.n_head, config.n_embd
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        outputs, weights = zip(
            *[h(x) for h in self.heads]
        )  # Unzip the outputs and weights
        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, torch.stack(
            weights, dim=1
        )  # Stack the attention weights from each head


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config: Config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        out, attn_weights = self.sa(
            self.ln1(x)
        )  # Receive the output and attention weights
        x = x + out  # Use only the output tensor for the residual connection
        x = x + self.ffwd(self.ln2(x))
        return x, attn_weights


class TransformerSequence(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

    def forward(self, x):
        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            all_attn_weights.append(attn_weights)
        return x, all_attn_weights
