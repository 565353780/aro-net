from aro_net.Module.gradio_server import GradioServer


def demo():
    port = 6002

    gradio_server = GradioServer(port)

    gradio_server.start()
    return True
