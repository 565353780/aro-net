import os
import torch
import numpy as np
from torch.utils.data import Dataset

from aro_net.Config.config import MASH_CONFIG
from aro_net.Method.feature import toMashFileAnchorFeature


class MashDataset(Dataset):
    def __init__(self, split) -> None:
        self.split = split
        self.n_qry = MASH_CONFIG.n_qry
        self.dir_dataset = os.path.join(MASH_CONFIG.dir_data, MASH_CONFIG.name_dataset)
        self.n_pts_train = MASH_CONFIG.n_pts_train
        self.n_pts_val = MASH_CONFIG.n_pts_val
        self.n_pts_test = MASH_CONFIG.n_pts_test
        self.files = []

        if split == "train":
            categories = MASH_CONFIG.categories_train
        else:
            # FIXME: only finish single class split here
            categories = MASH_CONFIG.categories_train

        for category in categories:
            if self.split == "train":
                id_shapes = (
                    open(f"{self.dir_dataset}/04_splits/{category}/mash/train.lst")
                    .read()
                    .split()
                )
            else:
                id_shapes = (
                    open(f"{self.dir_dataset}/04_splits/{category}/mash/test.lst")
                    .read()
                    .split()
                )

            for shape_id in id_shapes:
                qry_file_path = f"{self.dir_dataset}/02_qry_pts_occnet/{category}/{shape_id[:-4]}.npy"
                occ_file_path = f"{self.dir_dataset}/03_qry_occs_occnet/{category}/{shape_id[:-4]}.npy"

                if not os.path.exists(qry_file_path):
                    continue
                if not os.path.exists(occ_file_path):
                    continue

                self.files.append((category, shape_id))
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        category, full_shape_id = self.files[index]
        shape_id = full_shape_id[:-4]

        qry = np.load(f"{self.dir_dataset}/02_qry_pts_occnet/{category}/{shape_id}.npy")
        occ = np.load(
            f"{self.dir_dataset}/03_qry_occs_occnet/{category}/{shape_id}.npy"
        )

        mash_params_file_path = (
            f"{self.dir_dataset}/mash/{category}/{full_shape_id}.npy"
        )

        mash_params = np.load(mash_params_file_path, allow_pickle=True).item()

        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]
        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]

        params = np.hstack([mask_params, sh_params, rotate_vectors, positions])

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

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

        ftrs = toMashFileAnchorFeature(mash_params_file_path, qry, "cpu").numpy()

        n_ftrs = ftrs.shape[0]
        params = np.tile(params[np.newaxis, :, :], reps=(n_ftrs, 1, 1))
        ftrs = np.concatenate((params, ftrs), axis=-1)

        feed_dict = {
            "qry": torch.tensor(qry).float(),
            "ftrs": torch.tensor(ftrs).float(),
            "occ": torch.tensor(occ).float(),
        }

        if self.split != "train":
            feed_dict["shape_id"] = shape_id

        return feed_dict
