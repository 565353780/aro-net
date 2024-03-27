import os
import torch
import trimesh
from tqdm import tqdm

from aro_net.Config.config import get_parser
from aro_net.Model.mash import MashNet
from aro_net.Dataset.mash import MashDataset
from aro_net.Module.Generator3D.mash import Generator3D


def demo():
    device = "cpu"

    args = get_parser().parse_args()

    n_anc = 40  # FIXME
    model = MashNet(
        n_anc=n_anc,
        n_qry=args.n_qry,
        n_local=args.n_local,
        cone_angle_th=args.cone_angle_th,
        tfm_pos_enc=args.tfm_pos_enc,
        cond_pn=args.cond_pn,
        pn_use_bn=args.pn_use_bn,
        norm_coord=args.norm_coord,
    )
    path_ckpt = os.path.join("experiments", args.name_exp, "ckpt", args.name_ckpt)
    model.load_state_dict(torch.load(path_ckpt)["model"])
    model = model.to(device)
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
    )
    # dataset = ARONetDataset(split='test', args=args)
    dataset = MashDataset(split="asdf_test", args=args)  # FIXME

    dir_dataset = os.path.join(args.dir_data, args.name_dataset)
    if args.name_dataset == "shapenet":
        # categories = args.categories_test.split(',')[:-1]
        categories = ["02691156"]
        id_shapes = []
        for category in categories:
            print(category)
            id_shapes_ = (
                open(f"{dir_dataset}/04_splits/{category}/asdf_test.lst")
                .read()
                .split("\n")
            )
            id_shapes += id_shapes_

    else:
        id_shapes = open(f"{dir_dataset}/04_splits/test.lst").read().split("\n")

    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]
            for key in data:
                if key == "shape_id":
                    continue
                data[key] = data[key].unsqueeze(0).to(device)
            out = generator.generate_mesh(data)
            if isinstance(out, trimesh.Trimesh):
                mesh = out
            else:
                mesh = out[0]

            path_mesh = os.path.join(path_res, "%s.obj" % id_shapes[idx])
            mesh.export(path_mesh)

    return True
