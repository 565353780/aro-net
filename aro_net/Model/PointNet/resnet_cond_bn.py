import torch
import torch.nn as nn

from aro_net.Model.PointNet.common import maxpool
from aro_net.Model.Layer.c_resnet_block_conv1d import CResnetBlockConv1d


class ResnetPointnetCondBN(nn.Module):
    """PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(
        self, c_dim=128, dim=3, hidden_dim=128, reduce=True, norm_method="batch_norm"
    ):
        super().__init__()
        self.c_dim = c_dim
        self.reduce = reduce
        self.fc_pos = nn.Conv1d(dim, 2 * hidden_dim, 1)
        self.block_0 = CResnetBlockConv1d(
            c_dim=c_dim,
            size_in=2 * hidden_dim,
            size_h=hidden_dim,
            size_out=hidden_dim,
            norm_method=norm_method,
        )
        self.block_1 = CResnetBlockConv1d(
            c_dim=c_dim,
            size_in=2 * hidden_dim,
            size_h=hidden_dim,
            size_out=hidden_dim,
            norm_method=norm_method,
        )
        self.block_2 = CResnetBlockConv1d(
            c_dim=c_dim,
            size_in=2 * hidden_dim,
            size_h=hidden_dim,
            size_out=hidden_dim,
            norm_method=norm_method,
        )
        self.block_3 = CResnetBlockConv1d(
            c_dim=c_dim,
            size_in=2 * hidden_dim,
            size_h=hidden_dim,
            size_out=hidden_dim,
            norm_method=norm_method,
        )
        self.block_4 = CResnetBlockConv1d(
            c_dim=c_dim,
            size_in=2 * hidden_dim,
            size_h=hidden_dim,
            size_out=hidden_dim,
            norm_method=norm_method,
        )
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, c):
        # batch_size, T, D = p.size()
        p = p.permute(0, 2, 1)  # N, 4, N_point

        net = self.fc_pos(p)
        net = self.block_0(net, c)

        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())

        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net, c)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net, c)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net, c)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net, c)
        if self.reduce:
            net = self.pool(net, dim=2)

            c = self.fc_c(self.actvn(net))
        else:
            c = net
        return c
