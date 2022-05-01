from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from mingpt.model import GPTConfig, Block


class Embedding2D(nn.Module):
    def __init__(self, config):
        super(Embedding2D, self).__init__()
        self.pos_emb_shape = (config.n_tiles_per_axis, config.n_embd)
        self.h_embed = nn.Parameter(torch.randn(self.pos_emb_shape) * 0.02)
        self.v_embed = nn.Parameter(torch.randn(self.pos_emb_shape) * 0.02)

    def forward(self, x):
        emb_h = self.h_embed.view(1, self.pos_emb_shape[0], 1, self.pos_emb_shape[1])
        emb_v = self.v_embed.view(1, 1, self.pos_emb_shape[0], self.pos_emb_shape[1])
        return x + (emb_h + emb_v).view(1, -1, self.pos_emb_shape[1])


class ViT(nn.Module):
    def __init__(self, config):
        super(ViT, self).__init__()
        self.config = deepcopy(config)
        self.config.use_mask = False
        self.config.block_size = 64
        patch_numel = config.window_size ** 2 * 3
        self.patch_emb = nn.Linear(patch_numel, config.n_embd, bias=False)
        self.embedding2d = Embedding2D(config)
        self.block_list = nn.ModuleList()
        for _ in range(config.n_layer):
            self.block_list.append(Block(config))
        self.ln_f = nn.LayerNorm(config.n_embd)

    def _tile(self, x):
        """
        Performs tiling (tokenization) from a batch of images
        :param x: batch of raw unprocessed images: (bs, 64, 64, 3), torch.Tensor
        :return: tensor containing n_tiles * n_tiles patches: (bs, n_tiles**2, 8, 8, 3), last 3 dims not being reduced
        to allow a potential convolutional stem
        """
        ntpa = self.config.n_tiles_per_axis
        ws = self.config.window_size
        x = x.view(-1, ntpa, ws, ntpa, ws, 3)
        x = x.transpose(2, 3).contiguous()
        x = x.view(-1, ntpa * ntpa, ws, ws, 3)
        return x

    def forward(self, x):
        """
        Computes 2-dimensional logits per patch
        :param x: (bs, 64, 64, 3)
        :return: (bs, n_tiles * n_tiles, emb_dim)
        """
        x = self._tile(x)
        x = torch.flatten(x, start_dim=2, end_dim=-1)
        x = self.patch_emb(x)
        x = self.embedding2d(x)
        for block in self.block_list:
            x = block(x)
        x = self.ln_f(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


class DecisionTransformer(nn.Module):
    def __init__(self, config):
        super(DecisionTransformer, self).__init__()
        self.vit = ViT(config)
        self.rtg_projection = nn.Linear(1, config.n_embd, bias=False)
        self.action_projection = nn.Linear(10, config.n_embd, bias=False)
        self.pe = PositionalEmbedding(config.n_embd)

        self.dt_blocks = nn.ModuleList()
        for _ in range(config.n_layer):
            block = Block(config)
            self.dt_blocks.append(block)

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.action_reprojection = nn.Linear(config.n_embd, 10)


    def forward(self, rtgs, obss, actions, inds):
        bs, seq_len = obss.shape[:2]
        arange = torch.arange(*inds).to(obss.device)
        pe = self.pe(arange)
        obss = obss.view(bs * seq_len, *obss.shape[2:])
        vision_embed = self.vit(obss).mean(dim=-2)
        vision_embed = vision_embed.view(bs, seq_len, -1) + pe
        actions = self.action_projection(actions) + pe
        rtgs = self.rtg_projection(rtgs) + pe

        # Concatenation
        x = torch.stack([rtgs, vision_embed, actions], dim=2)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

        for block in self.dt_blocks:
            x = block(x)

        # Deconcatenation
        x = x.view(x.shape[0], x.shape[1] // 3, 3, x.shape[2])[:, :, 1, :]  # Single out state->action mapping
        x = self.ln_f(x)
        out = self.action_reprojection(x)

        return out


if __name__ == "__main__":
    n_tiles_per_axis = 8
    window_size = 8
    seq_len = 256

    config = GPTConfig(vocab_size=1,
                       block_size=seq_len * 3,
                       n_tiles_per_axis=n_tiles_per_axis,
                       window_size=8,
                       n_embd=128,
                       n_layer=6,
                       n_head=8,
                       seq_len=256)
    model = DecisionTransformer(config=config).cuda()
    rtgs = torch.zeros((1, seq_len, 1)).cuda()
    obss = torch.zeros((1, seq_len, 64, 64, 3)).cuda()
    actions = torch.zeros((1, seq_len, 10)).cuda()
    inds = (10, 266)

    out = model(rtgs, obss, actions, inds)

    inds = (30, 286)

    out2 = model(rtgs, obss, actions, inds)
    print(out == out2)
    print(out.shape)
