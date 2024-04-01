import os
import torch
import trimesh
import numpy as np
import open3d as o3d
from typing import Union

from aro_net.Config.config import ARO_CONFIG
from aro_net.Model.aro import ARONet
from aro_net.Method.sample import sampleQueryPoints
from aro_net.Module.Generator3D.aro import Generator3D


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
    ) -> None:
        self.n_anc = 48

        assert self.n_anc in [24, 48, 96]
        self.anc_0 = np.load(
            "../aro-net/aro_net/Data/anchors/sphere" + str(self.n_anc) + ".npy"
        )
        self.anc_np = np.concatenate([self.anc_0[i::3] / (2**i) for i in range(3)])
        self.anc = torch.from_numpy(self.anc_np).unsqueeze(0).to(ARO_CONFIG.device)

        self.model = ARONet()

        self.generator = Generator3D(self.model)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        state_dict = torch.load(model_file_path)["model"]

        # FIXME: an extra unused layer values occured
        remove_key_list = ["fc_dist_hit.0.weight", "fc_dist_hit.0.bias"]
        for remove_key in remove_key_list:
            if remove_key in state_dict.keys():
                del state_dict[remove_key]

        self.model.load_state_dict(state_dict)
        self.model.to(ARO_CONFIG.device)
        self.model.eval()
        return True

    @torch.no_grad()
    def detect(self, points: np.ndarray) -> trimesh.Trimesh:
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        center = (min_bound + max_bound) / 2.0

        scale = np.max(max_bound - min_bound)

        trans_points = (points - center) / scale

        query_points = sampleQueryPoints(trans_points, 512)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(trans_points)

        sample_pcd = pcd.farthest_point_down_sample(1024)
        sample_points = np.asarray(sample_pcd.points).astype(np.float32)

        # FIXME: qry is unused for current inference
        data = {
            "pcd": torch.from_numpy(sample_points).unsqueeze(0).to(ARO_CONFIG.device),
            "qry": torch.from_numpy(query_points).unsqueeze(0).to(ARO_CONFIG.device),
            "anc": self.anc,
        }

        out = self.generator.generate_mesh(data)

        if isinstance(out, trimesh.Trimesh):
            mesh = out
        else:
            mesh = out[0]

        mesh.vertices = mesh.vertices * scale + center

        return mesh

    @torch.no_grad()
    def detectFile(self, pcd_file_path: str) -> Union[trimesh.Trimesh, None]:
        if not os.path.exists(pcd_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t pcd file not exist!")
            print("\t pcd_file_path:", pcd_file_path)
            return None

        if pcd_file_path.endswith(".npy"):
            points = np.load(pcd_file_path)

            return self.detect(points)

        pcd = o3d.io.read_point_cloud(pcd_file_path)

        points = np.asarray(pcd.points)

        return self.detect(points)
