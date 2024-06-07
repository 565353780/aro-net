from aro_net.Demo.trainer import demo as demo_train
from aro_net.Demo.generator_3d import demo as demo_recon
from aro_net.Demo.detector import demo as demo_detect
from aro_net.Demo.gradio_server import demo as demo_gradio_server

if __name__ == "__main__":
    demo_train()
    demo_recon()
    demo_detect()
    demo_gradio_server()
