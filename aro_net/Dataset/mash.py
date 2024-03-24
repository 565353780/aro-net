import sys

sys.path.append("../ma-sh")

import os
import torch
import numpy as np
from torch.utils.data import Dataset


from ma_sh.Model.mash import Mash

from aro_net.Method.mesh import load_mesh


class MashDataset(Dataset):
    def __init__(self, split, args) -> None:
        # categories = args.categories_train.split(',')[:-1] # only check airplanes
        self.split = split
        # self.n_anc = args.n_anc
        self.n_anc = 40  # FIXME:
        self.n_qry = args.n_qry
        self.dir_dataset = os.path.join(args.dir_data, args.name_dataset)
        # self.anc_0 = np.load(f'./{args.dir_data}/anchors/sphere{str(self.n_anc)}.npy')
        # self.anc = np.concatenate([self.anc_0[i::3] / (2 ** i) for i in range(3)])
        self.name_dataset = args.name_dataset
        self.n_pts_train = args.n_pts_train
        self.n_pts_val = args.n_pts_val
        self.n_pts_test = args.n_pts_test
        self.gt_source = args.gt_source
        self.files = []
        # if self.name_dataset == 'shapenet':
        #     if self.split in {'train', 'val'}:
        #         categories = args.categories_train.split(',')[:-1]
        #     else:
        #         categories = args.categories_test.split(',')[:-1]
        #     self.fext_mesh = 'obj'
        # else:
        #     categories = ['']
        #     self.fext_mesh = 'ply'

        categories = ["02691156"]  # FIXME
        for category in categories:
            if self.split == "train":
                id_shapes = (
                    open(f"{self.dir_dataset}/04_splits/{category}/asdf.lst")
                    .read()
                    .split()
                )
                for shape_id in id_shapes:
                    self.files.append((category, shape_id))
            else:
                id_shapes = (
                    open(f"{self.dir_dataset}/04_splits/{category}/asdf_test.lst")
                    .read()
                    .split()
                )
                for shape_id in id_shapes:
                    self.files.append((category, shape_id))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        category, shape_id = self.files[index]
        # Only consider shapenet dataset: when do training and validation, we use the pcd, qry, occ provided by occ-net or im-net;
        if self.split == "train":
            if self.name_dataset == "shapenet":
                pass
                # pcd = np.load(f'{self.dir_dataset}/01_pcds/{category}/occnet/{shape_id}.npy')
                # np.random.seed()
                # perm = np.random.permutation(len(pcd))[:self.n_pts_train]
                # pcd = pcd[perm]
            else:
                mesh = load_mesh(
                    f"{self.dir_dataset}/00_meshes/{category}/{shape_id}.{self.fext_mesh}"
                )
                pcd = mesh.sample(self.n_pts_train)
        else:
            pcd = np.load(
                f"{self.dir_dataset}/01_pcds/{category}/{str(self.n_pts_test)}/{shape_id}.npy"
            )

        qry = np.load(
            f"{self.dir_dataset}/02_qry_pts_{self.gt_source}/{category}/{shape_id}.npy"
        )
        occ = np.load(
            f"{self.dir_dataset}/03_qry_occs_{self.gt_source}/{category}/{shape_id}.npy"
        )
        ftrs = np.load(
            f"{self.dir_dataset}/06_asdf_params_right/{category}/{shape_id}.npy"
        )
        anc_param_path = f"{self.dir_dataset}/05_asdf_anchor/{category}/{shape_id}.npy"
        anc_params = np.load(anc_param_path, allow_pickle=True).item()
        params = anc_params["best_params"]
        n_ftrs = ftrs.shape[0]
        # print(ftrs.shape)
        params = np.tile(params[np.newaxis, :, :], reps=(n_ftrs, 1, 1))
        # print(params.shape)
        ftrs = np.concatenate((params, ftrs), axis=-1)

        sdf = occ

        if self.split == "train":
            np.random.seed()
            perm = np.random.permutation(len(qry))[: self.n_qry]
            perm = np.array(list(range(self.n_qry)))
            qry = qry[perm]
            occ = occ[perm]
            sdf = sdf[perm]
            # ftrs = self.get_anchor_feature(qry, anc_param_path)
            ftrs = ftrs[perm]
            # params = parmas[perm, :]
        else:
            np.random.seed(1234)
            perm = np.random.permutation(len(qry))[: self.n_qry]
            qry = qry[perm]
            occ = occ[perm]
            sdf = sdf[perm]
            ftrs = ftrs[perm]
            # params = parmas[perm, :]

        feed_dict = {
            # 'pcd': torch.tensor(pcd).float(),
            "qry": torch.tensor(qry).float(),
            "ftrs": torch.tensor(ftrs).float(),
            # 'anc': torch.tensor(self.anc).float(),
            "occ": torch.tensor(occ).float(),
            # 'shape_id': shape_id,
            # 'params':torch.tensor(params).float(),
        }
        if self.split != "train":
            feed_dict["shape_id"] = shape_id

        return feed_dict
