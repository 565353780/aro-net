import numpy as np

from aro_net.Method.feature import get_anchor_feature


def demo():
    #  npy_path = "./data/shapenet/05_asdf_anchor/02691156/1a9b552befd6306cc8f2d5fe7449af61.npy"
    #  qry = np.load(f"./data/shapenet/02_qry_pts_imnet/02691156/1a9b552befd6306cc8f2d5fe7449af61.npy")

    lst_path = "./data/shapenet/04_splits/02691156/asdf.lst"
    npy_path_base = "./data/shapenet/05_asdf_anchor/02691156/"
    qry_path_base = "./data/shapenet/02_qry_pts_imnet/02691156/"

    save_path_base = "./data/shapenet/06_asdf_params_right/02691156/"

    with open(lst_path, "r") as file:
        for line in file:
            shape_id = line.strip()
            print(f"processing file {shape_id}...")

            npy_path = npy_path_base + str(shape_id) + ".npy"
            qry_path = qry_path_base + str(shape_id) + ".npy"
            qry = np.load(qry_path)

            anc_ftrs = get_anchor_feature(qry, npy_path)

            save_path = save_path_base + str(shape_id) + ".npy"
            np.save(save_path, anc_ftrs)

    return True
