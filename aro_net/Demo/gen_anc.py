import argparse

from aro_net.Method.anchor import construct_anchors


def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, default="fibo", choices=["fibo", "grid", "uniform"]
    )
    parser.add_argument("--n_anc", type=int, default=48)
    parser.add_argument("--path_save", type=str, default="./data/anchors/")
    args = parser.parse_args()
    construct_anchors(args)
    return True
