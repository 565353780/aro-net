import trimesh
import numpy as np


def fibonacci_sphere(n=48, offset=False):
    """Sample points on sphere using fibonacci spiral.

    # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

    :param int n: number of sample points, defaults to 48
    :param bool offset: set True to get more uniform samplings when n is large , defaults to False
    :return array: points samples
    """

    golden_ratio = (1 + 5**0.5) / 2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / golden_ratio

    if offset:
        if n >= 600000:
            epsilon = 214
        elif n >= 400000:
            epsilon = 75
        elif n >= 11000:
            epsilon = 27
        elif n >= 890:
            epsilon = 10
        elif n >= 177:
            epsilon = 3.33
        elif n >= 24:
            epsilon = 1.33
        else:
            epsilon = 0.33
        phi = np.arccos(1 - 2 * (i + epsilon) / (n - 1 + 2 * epsilon))
    else:
        phi = np.arccos(1 - 2 * (i + 0.5) / n)

    x = np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1
    )
    return x


def get_patch_radius(grid_res: int = 256, epsilon: float = 3.0):
    return (1.0 + epsilon) / grid_res


def get_query_pts_for_mesh(
    in_mesh: trimesh.Trimesh,
    num_query_pts: int,
    patch_radius: float = get_patch_radius(),
    far_query_pts_ratio=0.1,
    rng=np.random.RandomState(),
):
    # assume mesh to be centered around the origin
    import trimesh.proximity

    def _get_points_near_surface(mesh: trimesh.Trimesh):
        samples, face_id = mesh.sample(num_query_pts_close, return_index=True)
        offset_factor = (
            (rng.random(size=(num_query_pts_close,)) - 0.5) * 2.0 * patch_radius
        )
        sample_normals = mesh.face_normals[face_id]
        sample_normals_len = np.sqrt(np.linalg.norm(sample_normals, axis=1))
        sample_normals_len_broadcast = np.broadcast_to(
            np.expand_dims(sample_normals_len, axis=1), sample_normals.shape
        )
        sample_normals_normalized = sample_normals / sample_normals_len_broadcast
        offset_factor_broadcast = np.broadcast_to(
            np.expand_dims(offset_factor, axis=1), sample_normals.shape
        )
        noisy_samples = samples + offset_factor_broadcast * sample_normals_normalized
        return noisy_samples

    num_query_pts_far = int(num_query_pts * far_query_pts_ratio)
    num_query_pts_close = num_query_pts - num_query_pts_far

    in_mesh.fix_normals()
    query_pts_close = _get_points_near_surface(in_mesh)

    # add e.g. 10% samples that may be far from the surface
    query_pts_far = (rng.random(size=(num_query_pts_far, 3))) - 0.5

    query_pts = np.concatenate((query_pts_far, query_pts_close), axis=0)

    return query_pts
