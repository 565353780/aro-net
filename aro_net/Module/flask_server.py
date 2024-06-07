import os
import numpy as np
import open3d as o3d
from flask import Flask

from aro_net.Module.detector import Detector

app = Flask(__name__)


model_file_path = "./output/4_7080.ckpt"
detector = Detector(model_file_path)


@app.route("/hello")
def hello_world():
    return "Hello World"


@app.route("/toAROMesh")
def toAROMesh(
    input_pcd_file_path: str,
):
    print("input_pcd_file_path:", input_pcd_file_path)
    if not os.path.exists(input_pcd_file_path):
        print("[ERROR][flask_server::toAROMesh]")
        print("\t input pcd file not exist!")
        print("\t input_pcd_file_path:", input_pcd_file_path)
        return ""

    input_pcd_file_name = input_pcd_file_path.split("/")[-1]
    save_pcd_file_path = "./output/" + input_pcd_file_name.replace(".ply", ".obj")

    pcd = o3d.io.read_point_cloud(input_pcd_file_path)
    gt_points = np.asarray(pcd.points)

    mesh = detector.detect(gt_points)

    mesh.export(save_pcd_file_path)

    return save_pcd_file_path


class FlaskServer(object):
    def __init__(self, port: int) -> None:
        self.port = port
        return

    def start(self) -> bool:
        app.run(host="0.0.0.0", port=self.port, debug=False)
        return True
