from aro_net.Config.config import get_parser
from aro_net.Module.Trainer.mash import Trainer


def demo():
    args = get_parser().parse_args()

    trainer = Trainer()

    if args.mode == "train":
        trainer.train(args)
    else:
        trainer.test(args)
    return True
