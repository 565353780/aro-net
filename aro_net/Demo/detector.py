import numpy as np
import open3d as o3d

from aro_net.Module.detector import Detector


def demo():
    if False:
        model_file_path = "./output/aro-net-v3/4_7080.ckpt"
        pcd_file_path = (
            "./data/shapenet/01_pcds/02691156/1024/d1a887a47991d1b3bc0909d98a1ff2b4.npy"
        )

        points = np.load(pcd_file_path)

    if True:
        model_file_path = "./output/4_7080.ckpt"

        pcd_file_path = "../open3d-manage/output/input_pcd/airplane_pcd.ply"
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        points = np.asarray(pcd.points)

    detector = Detector(model_file_path)

    mesh = detector.detect(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("./output/test_pcd.ply", pcd)
    mesh.export("./output/test_mesh.obj")
    print(mesh)
    return True
