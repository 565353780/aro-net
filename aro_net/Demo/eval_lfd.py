from aro_net.Config.config import get_parser
from aro_net.Metric.lfd import eval, report


def demo():
    args = get_parser().parse_args()
    eval(args)
    report(args)
