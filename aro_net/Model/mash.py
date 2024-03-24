import torch.nn as nn

from aro_net.Model.positional_encoding_1d import PositionalEncoding1D
from aro_net.Model.PointNet.resnet import ResnetPointnet
from aro_net.Model.PointNet.resnet_cond_bn import ResnetPointnetCondBN


class MashNet(nn.Module):
    def __init__(
        self,
        n_anc,
        n_qry,
        n_local,
        cone_angle_th,
        tfm_pos_enc=True,
        cond_pn=True,
        pn_use_bn=True,
        pred_type="occ",
        norm_coord=False,
    ):
        super().__init__()
        # FIXME check if the anchor is same as asdf
        if n_anc != 40:
            print("[ASDFNetModel]: n_anc should be 100!")
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

        # the input ftrs size is n_anc x 6 for [phi, theta, dq, is_in_mask, dist_from_sh]
        self.fc_1 = nn.Sequential(
            nn.Conv1d(67, self.hidden_dim // 2, 1),  # FIXME
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
        if self.pred_type == "occ":
            # self.fc_out = nn.Conv1d(self.n_anc * self.hidden_dim, 1, 1)
            self.fc_out = (nn.Conv1d(self.hidden_dim, 1, 1),)
            self.fc_out = nn.Sequential(
                nn.Conv1d(
                    self.n_anc * self.hidden_dim, self.n_anc * self.hidden_dim // 2, 1
                ),
                nn.Tanh(),
                nn.Conv1d(self.n_anc * self.hidden_dim // 2, 1, 1),
                nn.Tanh(),
            )
        else:
            self.fc_out = nn.Sequential(nn.Conv1d(self.hidden_dim, 1, 1), nn.Tanh())

    def cast_cone(self, pcd, anc, qry):
        # TO be deleted
        return

    def cal_relatives(self, hit, anc, qry):
        # TO be deleted
        return

    def forward(self, feed_dict):
        qry, ftrs = feed_dict["qry"], feed_dict["ftrs"]
        n_bs, n_qry = qry.shape[0], qry.shape[1]
        self.n_qry = (
            n_qry  # when doing marching cube, the number of query points may change
        )

        # pcd, anc = [], []
        # # cast cone to capture local points (hit), and calculate observations from query points
        # hit = self.cast_cone(pcd, anc, qry)
        # feat_anc2qry, feat_qry2hit = self.cal_relatives(hit, anc, qry)

        # # run point net to calculate local features
        # feat_qry2hit_rs = feat_qry2hit.view(n_bs * self.n_qry * self.n_anc, self.n_local, -1)

        # if self.cond_pn:
        #     cond_anc = self.fc_cond_2(self.fc_cond_1(anc.permute(0, 2, 1)))
        #     cond_anc = cond_anc.permute(0, 2, 1).unsqueeze(1).expand(-1, self.n_qry, -1, -1).reshape(n_bs * self.n_qry * self.n_anc, -1)
        #     feat_local = self.point_net(feat_qry2hit_rs, cond_anc)
        # else:
        #     feat_local = self.point_net(feat_qry2hit_rs)

        # feat_local = feat_local.view(n_bs, self.n_qry, self.n_anc, -1)

        # # concat dir and dist from anc and merge them
        # feat_local_radial = torch.cat([feat_anc2qry.permute(0, 3, 2, 1), feat_local.permute(0, 3, 2, 1)], 1)

        # x = ftrs.view(n_bs, self.n_qry, self.n_anc, -1)
        # x = ftrs
        x = self.fc_1(ftrs.reshape(n_bs, -1, self.n_anc * self.n_qry))

        # x = self.fc_1(feat_local_radial.reshape(n_bs, -1, self.n_anc * self.n_qry))
        x = self.fc_2(x)
        x = x.view(n_bs, -1, self.n_anc, self.n_qry).permute(
            0, 3, 2, 1
        )  # n_bs, -1, n_anc, n_qry
        x = x.reshape(n_bs * self.n_qry, self.n_anc, -1)

        # apply positional encoding
        if self.tfm_pos_enc:
            x = x + self.pos_enc(x)

        x = self.att_decoder(x)

        # output the predicted occupancy
        x1 = x.view(n_bs, self.n_qry, self.n_anc, -1)
        x2 = x1.view(n_bs, self.n_qry, self.n_anc * self.hidden_dim).permute(0, 2, 1)
        pred = self.fc_out(x2).view(n_bs, self.n_qry)

        ret_dict = {}

        if self.pred_type == "occ":
            ret_dict["occ_pred"] = pred
        else:
            ret_dict["sdf_pred"] = pred

        return ret_dict
