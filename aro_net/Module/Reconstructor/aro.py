import os
import time
import math
import torch
import trimesh
import numpy as np
import torch.optim as optim

from torch import autograd
from tqdm import trange, tqdm

from aro_net.Config.config import get_parser

from aro_net.Lib import libmcubes
from aro_net.Lib.libmise import MISE
from aro_net.Lib.common import make_3d_grid
from aro_net.Lib.libsimplify import simplify_mesh

from aro_net.Model.aro import ARONet
from aro_net.Dataset.aro import ARONetDataset


if __name__ == "__main__":
    args = get_parser().parse_args()

    model = ARONet(
        n_anc=args.n_anc,
        n_qry=args.n_qry,
        n_local=args.n_local,
        cone_angle_th=args.cone_angle_th,
        tfm_pos_enc=args.tfm_pos_enc,
        cond_pn=args.cond_pn,
        use_dist_hit=args.use_dist_hit,
        pn_use_bn=args.pn_use_bn,
        pred_type=args.pred_type,
        norm_coord=args.norm_coord,
    )
    path_ckpt = os.path.join("experiments", args.name_exp, "ckpt", args.name_ckpt)
    model.load_state_dict(torch.load(path_ckpt)["model"])
    model = model.cuda()
    model = model.eval()

    path_res = os.path.join("experiments", args.name_exp, "results", args.name_dataset)
    if not os.path.exists(path_res):
        os.makedirs(path_res)

    generator = Generator3D(
        model,
        threshold=args.mc_threshold,
        resolution0=args.mc_res0,
        upsampling_steps=args.mc_up_steps,
        chunk_size=args.mc_chunk_size,
        pred_type=args.pred_type,
    )
    dataset = ARONetDataset(split="test", args=args)

    dir_dataset = os.path.join(args.dir_data, args.name_dataset)
    if args.name_dataset == "shapenet":
        categories = args.categories_test.split(",")[:-1]
        id_shapes = []
        for category in categories:
            id_shapes_ = (
                open(f"{dir_dataset}/04_splits/{category}/test.lst").read().split("\n")
            )
            id_shapes += id_shapes_

    else:
        id_shapes = open(f"{dir_dataset}/04_splits/test.lst").read().split("\n")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]
            for key in data:
                data[key] = data[key].unsqueeze(0).cuda()
            out = generator.generate_mesh(data)
            try:
                mesh, stats_dict = out
            except TypeError:
                mesh, stats_dict = out, {}

            path_mesh = os.path.join(path_res, "%s.obj" % id_shapes[idx])
            mesh.export(path_mesh)
