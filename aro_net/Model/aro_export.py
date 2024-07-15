import torch
import numpy as np
import torch.nn as nn
from math import cos

from aro_net.Config.config import ARO_CONFIG
from aro_net.Model.positional_encoding_1d import PositionalEncoding1D
from aro_net.Model.PointNet.resnet import ResnetPointnet


class ARONet(nn.Module):
    def __init__(
        self,
        n_anc=ARO_CONFIG.n_anc,
        n_qry=ARO_CONFIG.n_qry,
        n_local=ARO_CONFIG.n_local,
        cone_angle_th=ARO_CONFIG.cone_angle_th,
        pn_use_bn=ARO_CONFIG.pn_use_bn,
    ):
        super().__init__()
        self.n_anc = n_anc
        self.n_local = n_local
        self.n_qry = n_qry
        self.cone_angle_th = cone_angle_th
        self.point_net = ResnetPointnet(
            dim=4, reduce=True, size_aux=(n_anc, n_local), use_bn=pn_use_bn
        )
        self.fc_1 = nn.Sequential(
            nn.Conv1d(4 + 128, 128, 1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.fc_2 = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.att_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, batch_first=True
        )
        self.att_decoder = nn.TransformerEncoder(self.att_layer, num_layers=6)
        self.fc_out = nn.Conv1d(self.n_anc * 128, 1, 1)

    def cast_cone(self, pcd: torch.Tensor, anc: torch.Tensor, qry: torch.Tensor):
        """
        Using a cone to capture points in point cloud.
        Input:
        pcd: [n_bs, n_pts, 3]
        anc: [n_bs, n_anc, 3]
        qry: [n_bs, n_qry, 3]
        Return:
        hit_all: [n_bs, n_qry, n_anc, n_local, 3]
        """
        top_k = self.n_local
        th = np.pi / (self.cone_angle_th)

        vec_anc2qry = qry[:, None, :, :] - anc[:, :, None, :]
        mod_anc2qry = torch.linalg.norm(vec_anc2qry, dim=-1)[:, :, :, None]
        norm_anc2qry = vec_anc2qry / mod_anc2qry
        ray_anc2qry = torch.cat(
            [anc[:, :, None, :].expand(-1, -1, self.n_qry, -1), norm_anc2qry], -1
        )

        hit_all = []
        pcd_tile = pcd[:, None, :, :].expand(-1, self.n_qry, -1, -1)
        for idx_anc in range(self.n_anc):
            # first calculate the angle between anc2qry and anc2pts
            ray_anc2qry_ = ray_anc2qry[:, idx_anc, :, :]#torch.Size([1, 729, 6])
            vec_anc2pts_ = pcd[:, None, :, :] - ray_anc2qry_[:, :, None, :3]# torch.Size([1, 729, 676029, 3])
            # print(ray_anc2qry_.shape[1],'\n',vec_anc2pts_.shape[2])
            mod_anc2pts_ = torch.linalg.norm(vec_anc2pts_, dim=-1)
            norm_anc2pts_ = vec_anc2pts_ / mod_anc2pts_[:, :, :, None]
            norm_anc2qry_ = ray_anc2qry_[:, :, None, 3:]
            cos_value = (norm_anc2qry_ * norm_anc2pts_).sum(-1)
            # filter out those points are not in the cone
            flt_angle = cos_value <= cos(th)
            mod_anc2pts_[flt_angle] = torch.inf
            tmp = torch.topk(mod_anc2pts_, top_k, dim=-1, largest=False)
            idx_topk, vl_topk = tmp.indices, tmp.values
            hit_raw = torch.gather(
                pcd_tile, 2, idx_topk[:, :, :, None].expand(-1, -1, -1, 3)
            )
            flt_pts = (vl_topk == torch.inf).float()

            # padding those filtered-out points with query pints
            qry_rs = qry.unsqueeze(2).expand(-1, -1, top_k, -1)
            flt_pts_rs = flt_pts.unsqueeze(3).expand(-1, -1, -1, 3)
            hit = qry_rs * flt_pts_rs + hit_raw * (1 - flt_pts_rs)
            hit_all.append(hit.unsqueeze(2))

        hit_all = torch.cat(hit_all, 2)

        return hit_all

    def cal_relatives(self, hit: torch.Tensor, anc: torch.Tensor, qry: torch.Tensor):
        """
        Calculate the modulus and normals (if needed) from anc to query and query to hit points
        Input:
        hit: [n_bs, n_qry, n_anc, n_local, 3]
        anc: [n_bs, n_anc, 3]
        qry: [n_bs, n_qry, 3]
        Output:
        feat_anc2qry: [n_bs, n_qry, n_anc, 4]
        feat_qry2hit: [n_bs, n_qry, n_anc, n_local, 4]
        """

        vec_anc2qry = qry[:, :, None, :] - anc[:, None, :, :]
        mod_anc2qry = torch.linalg.norm(vec_anc2qry, dim=-1)[..., None]
        norm_anc2qry = torch.div(vec_anc2qry, mod_anc2qry.expand(-1, -1, -1, 3))
        feat_anc2qry = torch.cat([norm_anc2qry, mod_anc2qry], -1)
        vec_qry2hit = hit - qry.view(-1, self.n_qry, 1, 1, 3)
        mod_qry2hit = torch.linalg.norm(vec_qry2hit, dim=-1)[..., None]
        mask_padded = (mod_qry2hit.expand(-1, -1, -1, -1, 3) == 0).float()
        mod_qry2hit_ = (
            mod_qry2hit * (1 - mask_padded) + 1.0 * mask_padded
        )  # avoiding divide by 0
        norm_qry2hit = torch.div(vec_qry2hit, mod_qry2hit_)
        feat_qry2hit = torch.cat([norm_qry2hit, mod_qry2hit], -1)

        return feat_anc2qry, feat_qry2hit

    def forward(self, pcd: torch.Tensor, qry: torch.Tensor, anc: torch.Tensor):
        n_bs, n_qry = qry.shape[0], qry.shape[1]

        # when doing marching cube, the number of query points may change
        self.n_qry = n_qry

        # cast cone to capture local points (hit), and calculate observations from query points
        hit = self.cast_cone(pcd, anc, qry)
        feat_anc2qry, feat_qry2hit = self.cal_relatives(hit, anc, qry)

        # run point net to calculate local features
        feat_qry2hit_rs = feat_qry2hit.view(
            n_bs * self.n_qry * self.n_anc, self.n_local, -1
        )

        feat_local = self.point_net(feat_qry2hit_rs)

        feat_local = feat_local.view(n_bs, self.n_qry, self.n_anc, -1)

        # concat dir and dist from anc and merge them
        feat_local_radial = torch.cat(
            [feat_anc2qry.permute(0, 3, 2, 1), feat_local.permute(0, 3, 2, 1)], 1
        )

        x = self.fc_1(feat_local_radial.reshape(n_bs, -1, self.n_anc * self.n_qry))
        x = self.fc_2(x)
        x = x.view(n_bs, -1, self.n_anc, self.n_qry).permute(
            0, 3, 2, 1
        )  # n_bs, -1, n_anc, n_qry
        x = x.reshape(n_bs * self.n_qry, self.n_anc, -1)

        x = self.att_decoder(x)

        # output the predicted occupancy
        x1 = x.view(n_bs, self.n_qry, self.n_anc, -1)
        x2 = x1.view(n_bs, self.n_qry, self.n_anc * 128).permute(0, 2, 1)
        occ = self.fc_out(x2).view(n_bs, self.n_qry)

        return occ

    def forward_dict(self, feed_dict):
        pcd, qry, anc = feed_dict["pcd"], feed_dict["qry"], feed_dict["anc"]
        return self.forward(pcd, qry, anc)
