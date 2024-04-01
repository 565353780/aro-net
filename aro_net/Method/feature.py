import torch
import mash_cpp
import numpy as np
from typing import Union

from ma_sh.Model.mash import Mash


def toMashAnchorFeature(mash: Mash, query_points: torch.Tensor) -> torch.Tensor:
    """receive a qry numpy array, output the anchor feature from each anchor

    Returns:
       np.ndarray: num_qry x num_anchor x 5. 5 stands for the feature of each anchor: [theta, phi, d_q, 1/0, d].
       theta, phi is the spherical coordinate of the query point from the anchor;
       d_q is the distance between query point and the anchor;
       1/0 is whether the query point is within the  anchor's mask;
       d is the distance on the SH sphere, which is the distance between the anchor and the surface along the line of the query point and the anchor, d=0 if the query point is not within the mask.
    """

    ftrs = []
    for i in range(mash.anchor_num):
        inv_rotate_matrix = mash_cpp.toSingleRotateMatrix(-mash.rotate_vectors[i])

        local_rotate_query_points = query_points - mash.positions[i]

        local_query_points = torch.matmul(
            inv_rotate_matrix, local_rotate_query_points.reshape(-1, 3, 1)
        ).reshape(-1, 3)

        local_query_dists = torch.norm(local_query_points, p=2, dim=1)

        local_query_directions = local_query_points / local_query_dists.reshape(-1, 1)

        local_polars = mash_cpp.toPolars(local_query_directions)

        local_mask_boundary_thetas = mash_cpp.toSingleMaskBoundaryThetas(
            mash.mask_degree_max, mash.mask_params[i], local_polars[:, 0]
        )

        in_mask_mask = local_polars[:, 1] <= local_mask_boundary_thetas

        sh_dists = mash_cpp.toSingleSHDists(
            mash.sh_degree_max,
            mash.sh_params[i],
            local_polars[:, 0],
            local_polars[:, 1],
        )

        in_sh_dist_mask = local_query_dists <= sh_dists

        true_mask = in_mask_mask & in_sh_dist_mask

        # FIXME: need to set out of range point dists to 0?
        # sh_dists[~in_mask_mask] = 0.0

        label_diffs = local_rotate_query_points.type(torch.float64)
        label_polars = local_polars.type(torch.float64)
        label_dists = local_query_dists.reshape(-1, 1).type(torch.float64)
        label_in_mask = in_mask_mask.reshape(-1, 1).type(torch.float64)
        label_in_sh_dist = in_sh_dist_mask.reshape(-1, 1).type(torch.float64)
        label_in_cone = true_mask.reshape(-1, 1).type(torch.float64)
        label_sh_dists = sh_dists.reshape(-1, 1).type(torch.float64)

        ftr = torch.hstack(
            (
                label_diffs,
                label_polars,
                label_dists,
                label_in_mask,
                label_in_sh_dist,
                label_in_cone,
                label_sh_dists,
            )
        ).unsqueeze(0)
        ftrs.append(ftr)

    ftrs = torch.vstack(ftrs)
    ftrs = ftrs.permute(1, 0, 2)

    return ftrs


def toMashFileAnchorFeature(
    mash_params_file_path: str,
    query_points: Union[torch.Tensor, np.ndarray],
    device: str = "cpu",
) -> torch.Tensor:
    mask_boundary_sample_num = 0
    sample_polar_num = 0
    sample_point_scale = 0.0
    idx_dtype = torch.int64
    dtype = torch.float64

    if isinstance(query_points, np.ndarray):
        query_points = torch.from_numpy(query_points).type(dtype).to(device)

    mash = Mash.fromParamsFile(
        mash_params_file_path,
        mask_boundary_sample_num,
        sample_polar_num,
        sample_point_scale,
        idx_dtype,
        dtype,
        device,
    )

    return toMashAnchorFeature(mash, query_points)
