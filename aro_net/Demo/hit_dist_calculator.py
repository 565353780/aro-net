from aro_net.Config.config import get_parser
from aro_net.Module.hit_dist_calculator import HitDistCalculator


def demo():
    args = get_parser().parse_args()

    calculator = HitDistCalculator(args)
    calculator.cal_multi_processes()
    return True
