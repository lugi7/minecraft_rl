import torch
from torch import nn
from mingpt.model import GPTConfig, Block


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.pos_emb = nn.Parameter(0.02 * torch.randn((1, config.block_size, config.n_embd)))
        self.action_emb = nn.Linear(10, config.n_embd, bias=False)
        self.block_list = nn.ModuleList()
        for _ in range(config.n_layer):
            self.block_list.append(Block(config))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.out_projection = nn.Linear(config.n_embd, 10)

    def forward(self, diff, actions):
        bs = diff.shape[0]
        inp = diff.view(bs, 1, -1) + self.pos_emb
        inp[:, 1:] = inp[:, 1:] + self.action_emb(actions[:, :-1])
        x = inp
        for block in self.block_list:
            x = block(x)
        x = self.ln_f(x)
        return self.out_projection(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, block_layout: list[int]):
        super(ConvBlock, self).__init__()
        self.module_list = nn.ModuleList()
        in_channels = in_channels
        for n_filters in block_layout:
            self.module_list.append(nn.Conv2d(in_channels=in_channels,
                                              out_channels=n_filters,
                                              kernel_size=3,
                                              padding='same'))
            self.module_list.append(nn.ReLU())
            in_channels = n_filters
        self.module_list.append(nn.MaxPool2d(kernel_size=2))
        self.module_list.append(nn.BatchNorm2d(num_features=block_layout[-1]))

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, net_layout: list[list[int]], out_features: int):
        super(ConvNet, self).__init__()
        self.module_list = nn.ModuleList()
        in_channels = 3
        for block_layout in net_layout:
            self.module_list.append(ConvBlock(in_channels=in_channels,
                                              block_layout=block_layout))
            in_channels = block_layout[-1]
        self.module_list.append(nn.Flatten(start_dim=-3, end_dim=-1))

        flatten_dim = (64 // (2 ** len(net_layout))) ** 2 * in_channels
        self.module_list.append(nn.Linear(in_features=flatten_dim,
                                          out_features=out_features,
                                          bias=False))

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class StateActionMap(nn.Module):
    def __init__(self, state_features=256, n_actions=10, n_steps=1):
        super(StateActionMap, self).__init__()
        self.n_steps = n_steps
        self.conv_net = ConvNet(net_layout=[[32, 64], [64, 64], [128, 128], [256]], out_features=state_features)

        self.transformer_config = GPTConfig(vocab_size=1, block_size=n_steps, n_embd=128, n_layer=2, n_head=8)
        self.model = Transformer(self.transformer_config)


    def forward(self, x, actions):
        x = self.conv_net(x).view(2, x.shape[0] // 2, -1)
        x = torch.transpose(x, 0, 1)
        x = x[:, 1] - x[:, 0]
        x = self.model(x, actions)
        out_disc = torch.sigmoid(x[..., :-2])
        out_cont = torch.tanh(x[..., -2:])
        return out_disc, out_cont


if __name__ == '__main__':
    net = ConvNet(net_layout=[[32, 64], [128, 128], [128, 256], [256, 256]], out_features=256)
    inp = torch.zeros((1, 3, 64, 64))
    out = net(inp)
    print(out.shape)
    for x, y in net.named_parameters():
        print(x, y.data.numel())
    print(sum(x.numel() for x in net.parameters()))
