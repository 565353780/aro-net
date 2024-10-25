import os
import numpy as np
import open3d as o3d

from aro_net.Module.detector import Detector


if __name__ == "__main__":
    model_file_path = "./output/4_7080.ckpt"
    pcd_folder_path = "/home/chli/github/ASDF/conditional-flow-matching/output/sample/epoch106/"
    sample_point_num = 2048
    save_folder_path = "./output/CFM_epoch106/"

    if not os.path.exists(pcd_folder_path):
        print('folder not exist!')
        exit()

    detector = Detector(model_file_path)

    pcd_filename_list = os.listdir(pcd_folder_path)

    os.makedirs(save_folder_path, exist_ok=True)

    for i, pcd_filename in enumerate(pcd_filename_list):
        pcd_file_path = pcd_folder_path + pcd_filename
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
        print(mesh)
        print('finish convert', i + 1, '/', len(pcd_filename_list), '!')
