import os
import torch
import trimesh
from typing import Union

from aro_net.Config.config import MASH_CONFIG
from aro_net.Model.mash import MashNet
from aro_net.Module.Generator3D.mash import Generator3D


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
    ) -> None:
        self.model = MashNet()

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
        self.model.to(MASH_CONFIG.device)
        self.model.eval()
        return True

    @torch.no_grad()
    def detectFile(self, mash_params_file_path: str) -> Union[trimesh.Trimesh, None]:
        if not os.path.exists(mash_params_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t mash params file not exist!")
            print("\t mash_params_file_path:", mash_params_file_path)
            return None

        out = self.generator.generate_mesh(mash_params_file_path)

        if isinstance(out, trimesh.Trimesh):
            mesh = out
        else:
            mesh = out[0]

        return mesh
