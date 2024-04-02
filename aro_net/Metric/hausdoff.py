from scipy.spatial import distance


def eval_hausdoff(p1, p2):
    """p1: reconstructed points
    p2: reference ponits
    shapes: (N, 3)
    """
    dist_rec2ref, _, _ = distance.directed_hausdorff(p1, p2)
    dist_ref2rec, _, _ = distance.directed_hausdorff(p2, p1)
    dist = max(dist_rec2ref, dist_ref2rec)
    return dist_rec2ref, dist_ref2rec, dist
