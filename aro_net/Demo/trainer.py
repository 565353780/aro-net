import sys

sys.path.append("../ma-sh")

from aro_net.Module.trainer import Trainer


def demo():
    model_file_path = "./output/4_7080.ckpt"
    #model_file_path = None

    trainer = Trainer(model_file_path)

    trainer.train()
    return True
