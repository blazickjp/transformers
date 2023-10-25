import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.config import Config
from utils.d_set import DSet
from torch.utils.tensorboard import SummaryWriter
from models.transformer import TransformerSequence

config = Config()
torch.manual_seed(1337)
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
dataset = DSet("data/shakespeare.txt", config)
# Check if a GPU is available and if so, set the device to GPU
device = "cuda" if torch.cuda.is_available() else config.device
print(f"Using {device} device")

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter("runs/experiment_1")


class GPTLanguageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(dataset.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = TransformerSequence(config)
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, dataset.vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        x, attn_weights = self.blocks(x)  # Receive the output and attention weights
        x = self.ln_f(x)  # Apply layer normalization to the output tensor
        logits = self.lm_head(x)  # Compute logits from the output tensor

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, attn_weights  # Optionally return the attention weights

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last config.block_size tokens
            idx_cond = idx[:, -config.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def estimate_loss(self, dataset, eval_iters):
        out = {}
        self.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = dataset.get_batch(split)
                logits, loss, _ = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out


model = GPTLanguageModel(config=config)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

for iter in range(config.max_iters):
    # sample a batch of data
    xb, yb = dataset.get_batch("train")

    # evaluate the loss
    logits, loss, attn_weights = model(xb, yb)

    # every once in a while evaluate the loss on train and val sets
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        for layer_idx, layer_attn in enumerate(attn_weights):
            for head_idx, head_attn in enumerate(layer_attn):
                writer.add_image(
                    f"Attention/Layer{layer_idx + 1}/Head{head_idx + 1}",
                    head_attn[0],
                    iter,
                    dataformats="HW",
                )

        losses = model.estimate_loss(dataset, config.eval_iters)
        writer.add_scalar("Loss/train", losses["train"], iter)
        writer.add_scalar("Loss/val", losses["val"], iter)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
