"""Shared table attention building blocks."""
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class AttnPool(nn.Module):

    def __init__(self, dim, num_heads=4):
        super(AttnPool, self).__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.query)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x, mask):
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        attn_output, _ = self.attn(query, x, x, key_padding_mask=mask, need_weights=False)
        return attn_output.squeeze(1)

class SparseFeatureExtractor(nn.Module):

    def __init__(self, input_dim: int=23, embed_dim: int=64, num_heads: int=4, num_layers: int=2, mlp_hidden: int=128, output_dim: int=256, dropout: float=0.1, token_dropout: float=0.1):
        super().__init__()
        self.index_emb = nn.Embedding(input_dim, embed_dim)
        self.value_proj = nn.Linear(1, embed_dim)
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttnPool(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, output_dim))
