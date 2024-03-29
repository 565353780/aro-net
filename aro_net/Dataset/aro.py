import os
import torch
import numpy as np
from torch.utils.data import Dataset

from aro_net.Config.config import ARO_CONFIG


class ARONetDataset(Dataset):
    def __init__(self, split) -> None:
        self.split = split
        self.n_anc = ARO_CONFIG.n_anc
        self.n_qry = ARO_CONFIG.n_qry
        self.dir_dataset = os.path.join(ARO_CONFIG.dir_data, ARO_CONFIG.name_dataset)
        self.anc_0 = np.load(
            f"./{ARO_CONFIG.dir_data}/anchors/sphere{str(self.n_anc)}.npy"
        )
        self.anc = np.concatenate([self.anc_0[i::3] / (2**i) for i in range(3)])
        self.name_dataset = ARO_CONFIG.name_dataset
        self.n_pts_train = ARO_CONFIG.n_pts_train
        self.n_pts_val = ARO_CONFIG.n_pts_val
        self.n_pts_test = ARO_CONFIG.n_pts_test
        self.files = []
        if self.name_dataset == "shapenet":
            if self.split in {"train", "val"}:
                categories = ARO_CONFIG.categories_train
            else:
                categories = ARO_CONFIG.categories_test
            self.fext_mesh = "obj"
        else:
            categories = [""]
            self.fext_mesh = "ply"
        for category in categories:
            split_file_path = (
                self.dir_dataset + "/04_splits/" + category + "/" + split + ".lst"
            )
            assert os.path.exists(split_file_path), split_file_path + " not exist!"
            with open(split_file_path, "r") as f:
                id_shapes = f.read().split()

            for shape_id in id_shapes:
                self.files.append((category, shape_id))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        category, shape_id = self.files[index]

        if self.split == "train":
            pcd = np.load(
                f"{self.dir_dataset}/01_pcds/{category}/occnet/{shape_id}.npy"
            )
            np.random.seed()
            perm = np.random.permutation(len(pcd))[: self.n_pts_train]
            pcd = pcd[perm]
        elif self.split == "val":
            pcd = np.load(
                f"{self.dir_dataset}/01_pcds/{category}/occnet/{shape_id}.npy"
            )
            np.random.seed(1234)
            perm = np.random.permutation(len(pcd))[: self.n_pts_val]
            pcd = pcd[perm]
        else:
            pcd = np.load(
                f"{self.dir_dataset}/01_pcds/{category}/{str(self.n_pts_test)}/{shape_id}.npy"
            )

        qry = np.load(f"{self.dir_dataset}/02_qry_pts_occnet/{category}/{shape_id}.npy")
        occ = np.load(
            f"{self.dir_dataset}/03_qry_occs_occnet/{category}/{shape_id}.npy"
        )

        if self.split == "train":
            np.random.seed()

            positive_occ_idxs = np.where(occ > 0.5)[0]
            negative_occ_idxs = np.where(occ < 0.5)[0]

            positive_occ_num = self.n_qry // 2

            if positive_occ_num > positive_occ_idxs.shape[0]:
                positive_occ_num = positive_occ_idxs.shape[0]

            negative_occ_num = self.n_qry - positive_occ_num

            positive_idxs = np.random.choice(positive_occ_idxs, positive_occ_num)
            negative_idxs = np.random.choice(negative_occ_idxs, negative_occ_num)

            idxs = np.hstack([positive_idxs, negative_idxs])

            perm = idxs[np.random.permutation(self.n_qry)]

            qry = qry[perm]
            occ = occ[perm]
        else:
            np.random.seed(1234)
            perm = np.random.permutation(len(qry))[: self.n_qry]
            qry = qry[perm]
            occ = occ[perm]

        feed_dict = {
            "pcd": torch.tensor(pcd).float(),
            "qry": torch.tensor(qry).float(),
            "anc": torch.tensor(self.anc).float(),
            "occ": torch.tensor(occ).float(),
        }

        return feed_dict
