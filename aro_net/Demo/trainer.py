import sys

sys.path.append("../ma-sh")

from aro_net.Module.trainer import Trainer


def demo():
    trainer = Trainer()

    trainer.train()
    return True
