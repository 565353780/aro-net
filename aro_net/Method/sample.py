import trimesh
import numpy as np
import open3d as o3d
from typing import Union


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


def sampleNearSurfacePoints(
    mesh: trimesh.Trimesh,
    sample_point_num: int,
    patch_radius: float = get_patch_radius(),
    rng=np.random.RandomState(),
):
    samples, face_id = mesh.sample(sample_point_num, return_index=True)
    offset_factor = (rng.random(size=(sample_point_num,)) - 0.5) * 2.0 * patch_radius
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


def sampleMeshQueryPoints(
    in_mesh: trimesh.Trimesh,
    num_query_pts: int,
    patch_radius: float = get_patch_radius(),
    far_query_pts_ratio: float = 0.1,
    rng=np.random.RandomState(),
):
    num_query_pts_far = int(num_query_pts * far_query_pts_ratio)
    num_query_pts_close = num_query_pts - num_query_pts_far

    in_mesh.fix_normals()
    query_pts_close = sampleNearSurfacePoints(
        in_mesh, num_query_pts_close, patch_radius, rng
    )

    # add e.g. 10% samples that may be far from the surface
    query_pts_far = (rng.random(size=(num_query_pts_far, 3))) - 0.5

    query_pts = np.concatenate((query_pts_far, query_pts_close), axis=0)

    return query_pts


def samplePcdQueryPoints(
    points: np.ndarray,
    num_query_pts: int,
    patch_radius: float = get_patch_radius(),
    far_query_pts_ratio: float = 0.1,
    rng=np.random.RandomState(),
):
    num_query_pts_far = int(num_query_pts * far_query_pts_ratio)
    num_query_pts_close = num_query_pts - num_query_pts_far

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    fps_sample_pcd = pcd.farthest_point_down_sample(num_query_pts_close)
    fps_sample_points = np.asarray(fps_sample_pcd.points)

    gaussian_noise = np.random.normal(
        patch_radius / 2.0, patch_radius / 2.0, [num_query_pts_close, 3]
    )

    query_pts_close = fps_sample_points + gaussian_noise

    query_pts_far = (rng.random(size=(num_query_pts_far, 3))) - 0.5

    query_pts = np.concatenate((query_pts_far, query_pts_close), axis=0)

    return query_pts


def sampleQueryPoints(
    data: Union[trimesh.Trimesh, np.ndarray],
    num_query_pts: int,
    patch_radius: float = get_patch_radius(),
    far_query_pts_ratio: float = 0.1,
    rng=np.random.RandomState(),
) -> Union[np.ndarray, None]:
    if isinstance(data, trimesh.Trimesh):
        return sampleMeshQueryPoints(
            data, num_query_pts, patch_radius, far_query_pts_ratio, rng
        )

    if isinstance(data, np.ndarray):
        return samplePcdQueryPoints(
            data, num_query_pts, patch_radius, far_query_pts_ratio, rng
        )

    print("[ERROR][sample::sampleQueryPoints]")
    print("\t data instance not valid!")
    print("\t only support mesh[trimesh.Trimesh] and pcd[np.ndarray] as data!")
    return None
