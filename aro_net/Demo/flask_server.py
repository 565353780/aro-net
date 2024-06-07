from aro_net.Module.flask_server import FlaskServer


def demo():
    port = 9366

    flask_server = FlaskServer(port)
    flask_server.start()
    return True
