import torch
from torch import nn


class RAYTextEncoder(nn.Module):
    """Small trainable text encoder for the v0.1 prototype.

    v0.1 intentionally uses a simple whitespace tokenizer supplied by the
    training dataset. A stronger pretrained/frozen encoder can replace this
    module in a later release without changing the DiT interface.
    """

    def __init__(self, vocab_size=8192, dim=256, max_tokens=64, heads=4, depth=4):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, mask=None):
        x = self.token(tokens) + self.pos[:, :tokens.shape[1]]
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.norm(x)
