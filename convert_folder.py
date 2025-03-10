import os
import numpy as np
import open3d as o3d
from time import time

from aro_net.Module.detector import Detector


if __name__ == "__main__":
    model_file_path = "./output/pretrain-aro/4_7080.ckpt"
    pcd_folder_path = "/home/chli/chLi/Dataset/ShapeNet/manifold_pcd-8192_random-10/"
    sample_point_num = 8192
    save_folder_path = "./output/ShapeNet/"

    if not os.path.exists(pcd_folder_path):
        print('folder not exist!')
        exit()

    detector = Detector(model_file_path)

    os.makedirs(save_folder_path, exist_ok=True)

    i = 0
    for root, _, files in os.walk(pcd_folder_path):
        for file in files:
            if not file.endswith('.ply'):
                continue

            pcd_file_path = root + '/' + file

            start = time()
            pcd = o3d.io.read_point_cloud(pcd_file_path)

            sampled_pcd = pcd
            if len(pcd.points) > sample_point_num:
                sampled_pcd = pcd.farthest_point_down_sample(sample_point_num)

            points = np.asarray(sampled_pcd.points)

            mesh = detector.detect(points)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(save_folder_path + "pcd_" + str(i) + ".ply", pcd)
            mesh.export(save_folder_path + "mesh_" + str(i) + ".obj")
            time_spend = time() - start
            print(mesh)
            i += 1
            print('finish convert', i, '! time:', time_spend)
