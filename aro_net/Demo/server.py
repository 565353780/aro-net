from aro_net.Module.server import Server


def demo():
    port = 6002

    server = Server(port)

    server.start()
    return True
