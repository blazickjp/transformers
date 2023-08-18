import torch
from torch import nn
from torch.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from models.positional_encoder import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        decoder_layers = TransformerDecoderLayer(
            ninp, nhead, nhid, dropout, batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
            self.src_mask = mask

        src = self.encoder(src) * torch.sqrt(torch.tensor(self.ninp).float())
        src = self.pos_encoder(src)
        output = self.transformer_decoder(memory=src, tgt_mask=self.src_mask, tgt=src)
        output = self.decoder(output)

        return output
