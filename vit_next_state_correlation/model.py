import torch
from torch import nn
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
        self.config = config
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
        return x


if __name__ == "__main__":
    n_tiles_per_axis = 8
    window_size = 8
    config = GPTConfig(vocab_size=1,
                       block_size=n_tiles_per_axis ** 2,
                       n_tiles_per_axis=n_tiles_per_axis,
                       window_size=8,
                       n_embd=128,
                       n_layer=6,
                       n_head=8)
    model = ViT(config=config)
    inp = torch.zeros((256, 64, 64, 3))
    out = model(inp)
    print(out.shape)
