import torch
import torch.nn as nn

from aro_net.Model.PointNet.common import maxpool
from aro_net.Model.Layer.resnet_block_fc import ResnetBlockFC


class ResnetPointnet(nn.Module):
    """PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(
        self,
        c_dim=128,
        dim=3,
        hidden_dim=128,
        reduce=True,
        size_aux=(48, 32),
        use_bn=True,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.reduce = reduce
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(
            2 * hidden_dim, hidden_dim, size_aux=size_aux, use_bn=use_bn
        )
        self.block_1 = ResnetBlockFC(
            2 * hidden_dim, hidden_dim, size_aux=size_aux, use_bn=use_bn
        )
        self.block_2 = ResnetBlockFC(
            2 * hidden_dim, hidden_dim, size_aux=size_aux, use_bn=use_bn
        )
        self.block_3 = ResnetBlockFC(
            2 * hidden_dim, hidden_dim, size_aux=size_aux, use_bn=use_bn
        )
        self.block_4 = ResnetBlockFC(
            2 * hidden_dim, hidden_dim, size_aux=size_aux, use_bn=use_bn
        )
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        # batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)
        if self.reduce:
            net = self.pool(net, dim=1)

            c = self.fc_c(self.actvn(net))
        else:
            c = net
        return c
