import torch
from torch import nn

# Transformer causal self-attention layer - done
# Position embedding
# R_dash, state merge
# State embedding ConvNet
# Latent action space sampler - done
# Temporal causal downsampling layer

# Surprise measurer
# Latent action space surprise predictor
# Latent action decoder


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()

    def forward(self, x):
        # Input size: (..., 64, 64, 3)
        # Output size: (..., state_dim)
        
        raise NotImplementedError


class LatentSampler(nn.Module):
    def __init__(self):
        super(LatentSampler, self).__init__()
        self.distribution = torch.distributions.normal.Normal(0., 1.)

    def forward(self, means, stds):
        # Input shape is (batch_size, seq_len, latent_dim) for both means and stds
        sample = self.distribution.sample(means.shape).to(means.device)
        return sample * stds + means


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self):
        raise NotImplementedError


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv_projection = nn.Linear(embed_dim, 3 * n_heads * self.head_dim, bias=False)  # Experiment with bias
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(embed_dim, embed_dim)  # Experiment with getting rid of this

    @staticmethod
    def _get_mask(seq_len):
        mask = 1. - torch.tril(torch.ones((seq_len, seq_len))).view(1, 1, seq_len, seq_len)
        mask = mask * float('inf')
        return mask

    def forward(self, x):
        bs, seq_len = x.shape[:2]
        qkv = self.qkv_projection(x).view(bs, seq_len, self.n_heads, -1)
        q, k, v = torch.split(qkv, self.head_dim, dim=-1)  # Each has shape: (bs, seq_len, n_heads, head_dim)
        attn_logits = torch.einsum('bihd,bjhd->bijh', q, k) / torch.sqrt(self.head_dim)
        mask = self._get_mask(attn_logits.shape[1])
        masked_logits = attn_logits - mask
        attn_distribution = torch.softmax(masked_logits, dim=2)
        attn_distribution = self.dropout(attn_distribution)  # Shape is (bs, seq_len, seq_len, n_heads)
        out = torch.einsum('bijh,bjhd->bihd', attn_distribution, v)  # Shape is (bs, seq_len, n_heads, head_dim)
        out = out.view(bs, seq_len, -1)
        out = self.dropout(self.out_projection(out))
        return out


class PointwiseFFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super(PointwiseFFN, self).__init__()
        self.l1 = nn.Linear(embed_dim, ffn_dim)
        self.l2 = nn.Linear(ffn_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, block_dim):
        super(TransformerBlock, self).__init__()
        self.attention_layer = AttentionLayer(embed_dim=block_dim, n_heads=8)
        self.ffn = PointwiseFFN(embed_dim=block_dim, ffn_dim=block_dim * 4)

    def forward(self, x):
        x = x + self.attention_layer(x)
        x = x + self.ffn(x)
        return x
