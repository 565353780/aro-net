import torch.nn as nn

from aro_net.Config.config import MASH_CONFIG
from aro_net.Model.positional_encoding_1d import PositionalEncoding1D
from aro_net.Model.PointNet.resnet import ResnetPointnet
from aro_net.Model.PointNet.resnet_cond_bn import ResnetPointnetCondBN


class MashNet(nn.Module):
    def __init__(
        self,
        n_anc=MASH_CONFIG.n_anc,
        n_qry=MASH_CONFIG.n_qry,
        n_local=MASH_CONFIG.n_local,
        cone_angle_th=MASH_CONFIG.cone_angle_th,
        tfm_pos_enc=MASH_CONFIG.tfm_pos_enc,
        cond_pn=MASH_CONFIG.cond_pn,
        pn_use_bn=MASH_CONFIG.pn_use_bn,
        pred_type=MASH_CONFIG.pred_type,
        norm_coord=MASH_CONFIG.norm_coord,
    ):
        super().__init__()
        # FIXME check if the anchor is same as asdf
        if n_anc != 100:
            print("[MashNet]: n_anc should be 100!")
            raise NotImplementedError

        self.hidden_dim = 128
        self.n_anc = n_anc
        self.n_local = n_local
        self.n_qry = n_qry
        self.cone_angle_th = cone_angle_th
        self.cond_pn = cond_pn
        self.pred_type = pred_type
        self.norm_coord = norm_coord
        if self.cond_pn:
            self.point_net = ResnetPointnetCondBN(dim=4, reduce=True)
        else:
            self.point_net = ResnetPointnet(
                dim=4, reduce=True, size_aux=(n_anc, n_local), use_bn=pn_use_bn
            )
        self.tfm_pos_enc = tfm_pos_enc
        if self.cond_pn:
            self.fc_cond_1 = nn.Sequential(
                nn.Conv1d(3, self.hidden_dim, 1),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
            )
            self.fc_cond_2 = nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
            )

        # the input ftrs size is n_anc x 5 for [phi, theta, dq, is_in_mask, dist_from_sh]
        self.fc_1 = nn.Sequential(
            nn.Conv1d(30, self.hidden_dim // 2, 1),  # FIXME
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim // 2, self.hidden_dim // 2, 1),  # FIXME
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, 1),  # FIXME
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )
        self.fc_2 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )
        if self.tfm_pos_enc:
            self.pos_enc = PositionalEncoding1D(self.hidden_dim)
        self.att_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=8, batch_first=True
        )
        self.att_decoder = nn.TransformerEncoder(self.att_layer, num_layers=6)
        self.fc_out = nn.Sequential(
            nn.Conv1d(
                self.n_anc * self.hidden_dim, self.n_anc * self.hidden_dim // 2, 1
            ),
            nn.Tanh(),
            nn.Conv1d(
                self.n_anc * self.hidden_dim // 2, self.n_anc * self.hidden_dim // 2, 1
            ),
            nn.Tanh(),
            nn.Conv1d(self.n_anc * self.hidden_dim // 2, 1, 1),
            nn.Tanh(),
        )

    def forward(self, feed_dict):
        qry, ftrs = feed_dict["qry"], feed_dict["ftrs"]
        n_bs, n_qry = qry.shape[0], qry.shape[1]
        # when doing marching cube, the number of query points may change
        self.n_qry = n_qry

        x = self.fc_1(ftrs.reshape(n_bs, -1, self.n_anc * self.n_qry))

        x = self.fc_2(x)

        # n_bs, -1, n_anc, n_qry
        x = x.view(n_bs, -1, self.n_anc, self.n_qry).permute(0, 3, 2, 1)

        x = x.reshape(n_bs * self.n_qry, self.n_anc, -1)

        # apply positional encoding
        if self.tfm_pos_enc:
            x = x + self.pos_enc(x)

        x = self.att_decoder(x)

        # output the predicted occupancy
        x1 = x.view(n_bs, self.n_qry, self.n_anc, -1)
        x2 = x1.view(n_bs, self.n_qry, self.n_anc * self.hidden_dim).permute(0, 2, 1)

        occ = self.fc_out(x2).view(n_bs, self.n_qry)

        return occ
