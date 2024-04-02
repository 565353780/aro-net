import sys

sys.path.append("../ma-sh")

import numpy as np
import open3d as o3d
from ma_sh.Model.mash import Mash
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.render import renderGeometries
from aro_net.Dataset.aro import ARONetDataset
from aro_net.Dataset.mash import MashDataset


def test():
    aro_dataset = ARONetDataset("test")
    mash_dataset = MashDataset("test")

    for aro_data in aro_dataset:
        print(aro_data.keys())
        break

    for mash_data in mash_dataset:
        print(mash_data.keys())
        break

    mash_data = mash_dataset[0]
    shape_id = mash_data["shape_id"]
    print(shape_id)

    category, full_shape_id = mash_dataset.files[0]
    shape_id = full_shape_id[:-4]
    print(full_shape_id)
    print(shape_id)

    mesh_file_path = f"{mash_dataset.dir_dataset}/00_meshes/{category}/{shape_id}.obj"
    qry_file_path = (
        f"{mash_dataset.dir_dataset}/02_qry_pts_occnet/{category}/{shape_id}.npy"
    )
    mash_file_path = f"{mash_dataset.dir_dataset}/mash/{category}/{full_shape_id}.npy"

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    qry = getPointCloud(np.load(qry_file_path))
    mash = Mash.fromParamsFile(mash_file_path, 18, 4000, 0.4)

    mash_pcd = getPointCloud(mash.toSamplePoints().detach().clone().cpu().numpy())

    renderGeometries([mesh, qry, mash_pcd])

    return True
