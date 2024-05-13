import os
import numpy as np
from time import sleep

from aro_net.Module.Detector.aro import Detector


def demo():
    print('start convert new data...')

    model_file_path = "./output/pretrain-aro/4_7080.ckpt"
    sample_point_num = 4000

    detector = Detector(model_file_path)

    first_solve_list = ['03001627', '02691156']
    for category_id in first_solve_list:
        compare_folder_path = '../ma-sh/output/metric_manifold_result_selected/ShapeNet/' + category_id + '/'

        first_solve_shape_ids = None
        if os.path.exists(compare_folder_path):
            first_solve_shape_ids = os.listdir(compare_folder_path)
        #first_solve_shape_ids = None

        dataset_folder_path = '/home/chli/chLi/Dataset/SampledPcd_Manifold/ShapeNet/' + category_id + '/'
        save_folder_path = '/home/chli/chLi/Dataset/ARONet_Manifold_Recon_' + str(sample_point_num) + '/ShapeNet/' + category_id + '/'
        os.makedirs(save_folder_path, exist_ok=True)

        solved_shape_names = os.listdir(save_folder_path)

        pcd_filename_list = os.listdir(dataset_folder_path)
        pcd_filename_list.sort()

        for i, pcd_filename in enumerate(pcd_filename_list):
            if pcd_filename[-4:] != '.npy':
                continue

            if first_solve_shape_ids is not None:
                if pcd_filename.split('.npy')[0] not in first_solve_shape_ids:
                    continue

            if pcd_filename.replace('.npy', '.obj') in solved_shape_names:
                continue

            pcd_file_path = dataset_folder_path + pcd_filename

            points = np.load(pcd_file_path)

            recon_mesh = detector.detect(points)

            recon_mesh.export(save_folder_path + pcd_filename.replace('.npy', '.obj'))

            print('category:', category_id, 'solved shape num:', i + 1)

    print('convert new data finished!')
    return True

if __name__ == "__main__":
    while True:
        demo()
        sleep(10)
