import numpy as np
import open3d as o3d

from aro_net.Module.Detector.aro import Detector


def demo():
    model_file_path = "./output/pretrain-aro/4_7080.ckpt"

    detector = Detector(model_file_path)

    pcd_file_path = "./output/input_pcd/airplane_pcd.ply"
    pcd_file_path = "../param-gauss-recon/output/chair-denoise3-20000.ply"
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    points = np.asarray(pcd.points)

    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    length = np.max(max_bound - min_bound)
    scale = 0.9 / length
    center = (min_bound + max_bound) / 2.0
    points = (points - center) * scale
    pcd.points = o3d.utility.Vector3dVector(points)

    #sample_pcd = pcd.farthest_point_down_sample(20000)
    points = np.asarray(pcd.points)

    mesh = detector.detect(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("./output/test_pcd.ply", pcd, write_ascii=True)
    mesh.export("./output/test_mesh.obj")
    print(mesh)

    return True
