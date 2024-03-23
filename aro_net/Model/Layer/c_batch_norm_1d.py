import torch.nn as nn


class CBatchNorm1d(nn.Module):
    """Conditional batch normalization layer class.
    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    """

    def __init__(self, c_dim, f_dim, norm_method="batch_norm"):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == "batch_norm":
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == "instance_norm":
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == "group_norm":
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        elif norm_method == "layer_norm":
            self.bn = nn.LayerNorm(f_dim, elementwise_affine=False)
        else:
            raise ValueError("Invalid normalization method!")
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert x.size(0) == c.size(0)
        assert c.size(1) == self.c_dim

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        net = self.bn(x)

        out = gamma * net + beta

        return out
