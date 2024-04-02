import numpy as np

from aro_net.Method.mesh import normalize_mesh, sample_point_cloud


def demo():
    path_src = "mesh.obj"
    path_dst = "mesh_normalized.obj"
    normalize_mesh(path_src, path_dst)

    point_cloud = sample_point_cloud(path_dst, n_pts=2048)
    np.save("point_cloud.npy", point_cloud)
    return True
