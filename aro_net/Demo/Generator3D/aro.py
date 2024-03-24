import os
import torch
import trimesh
from tqdm import tqdm

from aro_net.Config.config import ARO_CONFIG
from aro_net.Model.aro import ARONet
from aro_net.Dataset.aro import ARONetDataset
from aro_net.Module.Generator3D.aro import Generator3D


def demo():
    model_file_path = "/home/chli/Models/aro-net/aronet_chairs_gt_imnet.ckpt"
    save_result_folder_path = "./output/aro/"

    model = ARONet()

    state_dict = torch.load(model_file_path)["model"]

    # FIXME: an extra unused layer values occured
    remove_key_list = ["fc_dist_hit.0.weight", "fc_dist_hit.0.bias"]
    for remove_key in remove_key_list:
        if remove_key in state_dict.keys():
            del state_dict[remove_key]

    model.load_state_dict(state_dict)
    model = model.to(ARO_CONFIG.device)
    model = model.eval()

    os.makedirs(save_result_folder_path, exist_ok=True)

    generator = Generator3D(model)
    dataset = ARONetDataset(split="test")

    dir_dataset = os.path.join(ARO_CONFIG.dir_data, ARO_CONFIG.name_dataset)
    if ARO_CONFIG.name_dataset == "shapenet":
        categories = ARO_CONFIG.categories_test.split(",")[:-1]
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

            del data["qry"]
            del data["occ"]
            del data["sdf"]
            for key in data:
                data[key] = data[key].unsqueeze(0).to(ARO_CONFIG.device)
                print(key, data[key].shape)

            out = generator.generate_mesh(data)

            if isinstance(out, trimesh.Trimesh):
                mesh = out
            else:
                mesh = out[0]

            print(mesh)
            path_mesh = save_result_folder_path + str(id_shapes[idx]) + ".obj"
            mesh.export(path_mesh)
            return

    return True
