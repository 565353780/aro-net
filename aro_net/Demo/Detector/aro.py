import numpy as np
import open3d as o3d

from aro_net.Module.Detector.aro import Detector


def demo():
    model_file_path = "./output/pretrain-aro/4_7080.ckpt"

    detector = Detector(model_file_path)

    pcd_file_path = "./output/input_pcd/airplane_pcd.ply"
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    points = np.asarray(pcd.points)

    mesh = detector.detect(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("./output/test_pcd.ply", pcd)
    mesh.export("./output/test_mesh.obj")
    print(mesh)

    return True
