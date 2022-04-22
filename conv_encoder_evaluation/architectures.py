import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


def get_backbone(name, embed_dim):
    if name == 'mobilenet_pretrained':
        base = models.mobilenet_v3_small(pretrained=True).features
        model = nn.Sequential(base, nn.Flatten(), nn.Linear(576 * 4, embed_dim))

    elif name == 'mobilenet_random':
        base = models.mobilenet_v3_small(pretrained=False).features
        model = nn.Sequential(base, nn.Flatten(), nn.Linear(576 * 4, embed_dim))

    elif name == 'convnet':
        model = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4, padding=0), nn.ReLU(),
                              nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                              nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                              nn.Flatten(), nn.Linear(1024, 512))
    return model


class CLIPModel(nn.Module):
    def __init__(self, backbone_name, embed_dim=512):
        super(CLIPModel, self).__init__()
        self.backbone = get_backbone(backbone_name, embed_dim=embed_dim)
        self.k_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.5)

    def forward(self, x):
        # x comes in shape(bs, 2, 64, 64, 3)
        bs = x.shape[0]
        x = x.view(-1, 64, 64, 3).permute(0, 3, 1, 2)  # (bs * 2, 3, 64, 64)
        x = self.backbone(x).view(bs, 2, -1)  # (bs, 2, embed_dim)
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]
        x2 = self.k_projection(x2)
        x1 = x1 / x1.norm(dim=1, keepdim=True)
        x2 = x2 / x2.norm(dim=1, keepdim=True)
        logits = F.softplus(self.logit_scale) * x1 @ x2.t()  # Shape is (bs, bs)
        return logits


