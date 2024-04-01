import numpy as np
import open3d as o3d

from aro_net.Module.Detector.aro import Detector


def demo():
    model_file_path = "./output/aro-net-v1/6_16611.ckpt"
    model_file_path = "./output/4_7080.ckpt"
    pcd_file_path = (
        "./data/shapenet/01_pcds/02691156/1024/d1a887a47991d1b3bc0909d98a1ff2b4.npy"
    )

    pcd_file_path = "./output/input_pcd/airplane_pcd.ply"

    # points = np.load(pcd_file_path)

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
