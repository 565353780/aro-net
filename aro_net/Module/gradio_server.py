import os
import numpy as np
import gradio as gr
import open3d as o3d

from aro_net.Method.render import toPlotFigure
from aro_net.Module.detector import Detector

model_file_path = "./output/4_7080.ckpt"
detector = Detector(model_file_path)


def renderInputData(input_pcd_file_path: str):
    pcd = o3d.io.read_point_cloud(input_pcd_file_path)

    gt_points = np.asarray(pcd.points)

    return toPlotFigure(gt_points)


def toAROMesh(
    input_pcd_file_path: str,
):
    print("input_pcd_file_path:", input_pcd_file_path)
    if not os.path.exists(input_pcd_file_path):
        print("[ERROR][gradio_server::toAROMesh]")
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


class GradioServer(object):
    def __init__(self, port: int) -> None:
        self.port = port
        return

    def start(self) -> bool:
        example_folder_path = "./output/input_pcd/"
        example_file_name_list = []
        if os.path.exists(example_folder_path):
            example_file_name_list = os.listdir(example_folder_path)

        examples = [
            example_folder_path + example_file_name
            for example_file_name in example_file_name_list
        ]

        with gr.Blocks() as iface:
            gr.Markdown("ARO-Net Inference Demo")

            with gr.Row():
                with gr.Column():
                    input_pcd = gr.Model3D(label="3D Data to be fitted")

                    gr.Examples(examples=examples, inputs=input_pcd)

                    submit_button = gr.Button("Submit to server")

                with gr.Column():
                    visual_gt_plot = gr.Plot()

                    recon_button = gr.Button("Click to start reconstructing")

            output_mesh = gr.Model3D(label="Reconstructed Surface")

            submit_button.click(
                fn=renderInputData,
                inputs=[input_pcd],
                outputs=[visual_gt_plot],
            )

            recon_button.click(
                fn=toAROMesh,
                inputs=[input_pcd],
                outputs=[output_mesh],
            )

        iface.launch(
            server_name="0.0.0.0",
            server_port=self.port,
            ssl_keyfile="./ssl/key.pem",
            ssl_certfile="./ssl/cert.pem",
            ssl_verify=False,
        )
        return True
