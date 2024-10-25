import torch.nn as nn


class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(
        self, size_in, size_out=None, size_h=None, size_aux=(48, 32), use_bn=True
    ):
        super().__init__()
        # Attributes
        self.use_bn = use_bn
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        self.n_anc = size_aux[0]
        self.n_local = size_aux[1]

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.n_anc * self.size_h)
        else:
            self.bn = nn.ModuleList()

    def perform_bn(self, net):
        n_dim = net.shape[-1]
        net = (
            net.view(-1, self.n_anc, self.n_local, n_dim)
            .permute(0, 1, 3, 2)
            .reshape(-1, self.n_anc * n_dim, self.n_local)
        )
        net = self.bn(net)
        net = (
            net.view(-1, self.n_anc, n_dim, self.n_local)
            .permute(0, 1, 3, 2)
            .reshape(-1, self.n_local, n_dim)
        )
        return net

    def forward(self, x):
        net = self.fc_0(self.actvn(x))  # B * N * M, n_local, size_h
        #FIXME: here can not be used by torch.jit.script
        # if self.use_bn:
        #     net = self.perform_bn(net)
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
