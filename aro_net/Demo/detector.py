import os
from shutil import copyfile

from aro_net.Method.time import getCurrentTime
from aro_net.Module.detector import Detector


def demo():
    model_file_path = "./output/pretrain-aro/4_7080.ckpt"
    device = 'cpu'
    save_folder_path = './output/recon/' + getCurrentTime() + '/'

    os.makedirs(save_folder_path, exist_ok=True)

    detector = Detector(model_file_path, device)

    pcd_file_path = "/home/chli/github/ASDF/td-shape-to-vec-set/output/sample/20241228_14:43:22/iter_18/category/03001627/pcd/sample_1_pcd.ply"

    mesh = detector.detectFile(pcd_file_path)
    print(mesh)
    mesh.export(save_folder_path + "recon_mesh.obj")

    copyfile(pcd_file_path, save_folder_path + "gt_pcd.ply")
    return True
