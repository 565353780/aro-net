import sys

sys.path.append("../ma-sh")

from aro_net.Module.trainer import Trainer


def demo():
    model_file_path = "./output/aro-net-v2-uniform_sample_occ/118_282387.ckpt"

    trainer = Trainer(model_file_path)

    trainer.train()
    return True
