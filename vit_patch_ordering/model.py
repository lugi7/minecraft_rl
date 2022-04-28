import torch
from torch import nn
from mingpt.model import GPTConfig, Block


class ViT(nn.Module):
    def __init__(self, config):
        super(ViT, self).__init__()
        self.config = config
        patch_numel = config.window_size ** 2 * 3
        self.patch_emb = nn.Linear(patch_numel, config.n_embd, bias=False)
        self.block_list = nn.ModuleList()
        for _ in range(config.n_layer):
            self.block_list.append(Block(config))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.out_projection = nn.Linear(config.n_embd, 2 * config.n_tiles_per_axis)

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
        x = x.transpose(2, 3)
        x = x.view(-1, ntpa * ntpa, ws, ws, 3)
        return x

    def forward(self, x):
        """
        Computes 2-dimensional logits per patch
        :param x: (bs, 64, 64, 3)
        :return: (bs, n_tiles * n_tiles, n_tiles, 2) - since class probability masses have to be at the 2nd axis
        """
        x = torch.flatten(x, start_dim=2, end_dim=-1)
        x = self.patch_emb(x)
        for block in self.block_list:
            x = block(x)
        x = self.ln_f(x)
        x = self.out_projection(x)
        ntpa = self.config.n_tiles_per_axis
        return x.view(-1, ntpa**2, 2, ntpa).permute(0, 3, 1, 2)


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
