import sys

sys.path.append("../ma-sh")

import open3d as o3d

from ma_sh.Model.mash import Mash

from aro_net.Module.Detector.mash import Detector


def demo():
    model_file_path = "./output/mash-v1/8_12447.ckpt"
    mash_params_file_path = "/home/chli/Dataset/aro_net/data/shapenet/mash/100anc/mash/02691156/2af04ef09d49221b85e5214b0d6a7_obj.npy"

    detector = Detector(model_file_path)

    mesh = detector.detectFile(mash_params_file_path)

    mash = Mash.fromParamsFile(mash_params_file_path, device="cuda:0")
    mash_points = mash.toSamplePoints().detach().clone().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mash_points)
    o3d.io.write_point_cloud("./output/test_mash_pcd.ply", pcd)
    mesh.export("./output/test_mash_mesh.obj")
    print(mesh)
    return True
