import os
import torch
import trimesh
from tqdm import tqdm

from aro_net.Config.config import get_parser
from aro_net.Model.aro import ARONet
from aro_net.Dataset.aro import ARONetDataset
from aro_net.Module.Generator3D.aro import Generator3D


def demo():
    model_file_path = "/home/chli/Models/aro-net/aronet_chairs_gt_occnet.ckpt"
    save_result_folder_path = "./output/aro/"
    device = "cpu"

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
    model.load_state_dict(torch.load(model_file_path)["model"])
    model = model.to(device)
    model = model.eval()

    os.makedirs(save_result_folder_path, exist_ok=True)

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

            if isinstance(out, trimesh.Trimesh):
                mesh = out
            else:
                mesh = out[0]

            path_mesh = save_result_folder_path + str(id_shapes[idx]) + ".obj"
            mesh.export(path_mesh)

    return True
