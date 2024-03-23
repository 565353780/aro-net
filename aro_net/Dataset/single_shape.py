import torch
import trimesh
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


class SingleShapeDataset(Dataset):
    def __init__(self, split, args):
        super().__init__()
        self.split = split
        self.n_anc = args.n_anc
        self.n_qry = args.n_qry
        self.shape = args.name_single
        self.mesh = trimesh.load(f"./{args.dir_data}/single/{self.shape}.ply")
        self.vox = trimesh.load(f"./{args.dir_data}/single/{self.shape}.binvox")
        self.anc_0 = np.load(f"./{args.dir_data}/anchors/sphere{str(self.n_anc)}.npy")
        self.anc = np.concatenate([self.anc_0[i::3] / (2**i) for i in range(3)])
        self.n_pts_train = args.n_pts_train
        self.n_pts_val = args.n_pts_val
        self.n_pts_test = args.n_pts_test
        if split == "train":
            self.n_pts = self.n_pts_train
        elif split == "test":
            self.n_pts = self.n_pts_test
        elif split == "val":
            self.n_pts = self.n_pts_val

    def __len__(self):
        if self.split == "train":
            return 400
        else:
            return 64

    def __getitem__(self, _):
        mesh = self.mesh.copy()
        R = np.eye(4)
        rot = Rotation.random()
        R[:3, :3] = rot.as_matrix()
        mesh.apply_transform(R)

        scale = np.random.uniform(0.5, 1.5, 3)
        mesh.apply_scale(scale)

        box = mesh.bounding_box
        l = box.scale
        c = box.centroid

        mesh.vertices = (mesh.vertices - c) / l

        bounds = mesh.bounds + 0.5
        extents = mesh.extents
        margin = 1 - extents
        t = -bounds[0] + np.random.random(3) * margin
        mesh.vertices = mesh.vertices + t

        N_near = self.n_qry // 2
        N_far = self.n_qry - N_near
        pts_near = mesh.sample(N_near) + (np.random.rand(N_near, 3) - 0.5) / 16
        pts_far = np.random.random([N_far, 3]) - 0.5
        pts = np.concatenate([pts_near, pts_far])
        pts_before = rot.inv().apply(((pts - t) * l + c) / scale[None])
        occ = self.vox.is_filled(pts_before).astype("float32")

        pcd = mesh.sample(self.n_pts)

        feed_dict = {
            "pcd": torch.tensor(pcd).float(),
            "anc": torch.tensor(self.anc).float(),
            "qry": torch.tensor(pts).float(),
            "occ": torch.tensor(occ).float(),
        }
        return feed_dict
