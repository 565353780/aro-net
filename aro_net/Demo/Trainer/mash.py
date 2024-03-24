from aro_net.Config.config import get_parser
from aro_net.Module.Trainer.mash import Trainer


def demo():
    args = get_parser().parse_args()

    trainer = Trainer(args)

    trainer.train()
    return True
