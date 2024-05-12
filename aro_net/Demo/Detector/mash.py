import sys

sys.path.append("../ma-sh")

import open3d as o3d
from shutil import copyfile

from ma_sh.Model.mash import Mash

from aro_net.Module.Detector.mash import Detector


def demo():
    dataset_root_folder_path = "/home/chli/Dataset/aro_net/data/shapenet/"
    mash_params_folder_path = dataset_root_folder_path + "mash/100anc/mash/"
    mesh_folder_path = dataset_root_folder_path + "00_meshes/"

    model_id = "02691156/2af04ef09d49221b85e5214b0d6a7"

    model_file_path = "./output/mash-uniform_sample/116_161811.ckpt"
    mash_params_file_path = mash_params_folder_path + model_id + "_obj.npy"

    detector = Detector(model_file_path)

    mesh = detector.detectFile(mash_params_file_path)

    print(mesh)

    mesh.export("./output/test_mash_mesh.obj")

    mash = Mash.fromParamsFile(mash_params_file_path, device="cuda:0")
    mash_points = mash.toSamplePoints().detach().clone().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mash_points)
    o3d.io.write_point_cloud("./output/test_mash_pcd.ply", pcd)

    gt_mesh_file_path = mesh_folder_path + model_id + ".obj"
    copyfile(gt_mesh_file_path, "./output/test_mash_mesh_gt.obj")
    return True