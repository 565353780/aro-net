from scipy.spatial import cKDTree


def points_dist(p1, p2, k=1, return_ind=False):
    """distance from p1 to p2"""
    tree = cKDTree(p2)
    dist, ind = tree.query(p1, k=k)
    if return_ind:
        return dist, ind
    else:
        return dist


def chamfer_dist(p1, p2):
    d1 = points_dist(p1, p2) ** 2
    d2 = points_dist(p2, p1) ** 2
    return d1, d2


def eval_chamfer(p1, p2, f_thresh=0.01):
    """p1: reconstructed points
    p2: reference ponits
    shapes: (N, 3)
    """
    d1, d2 = chamfer_dist(p1, p2)

    d1sqrt, d2sqrt = (d1**0.5), (d2**0.5)
    chamfer_L1 = 0.5 * (d1sqrt.mean(axis=-1) + d2sqrt.mean(axis=-1))
    chamfer_L2 = 0.5 * (d1.mean(axis=-1) + d2.mean(axis=-1))

    precision = (d1sqrt < f_thresh).sum(axis=-1) / p1.shape[0]
    recall = (d2sqrt < f_thresh).sum(axis=-1) / p2.shape[0]
    fscore = 2 * (recall * precision / recall + precision)

    return [chamfer_L1, chamfer_L2, fscore, precision, recall]
