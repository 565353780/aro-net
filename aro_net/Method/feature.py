import sys

sys.path.append("../ma-sh")

import numpy as np
from ma_sh.Model.Mash import Mash


def get_anchor_feature(qry: np.array, anc_param_path: str) -> np.array:
    """receive a qry numpy array, output the anchor feature from each anchor

    Args:
       qry (np.array): query points np array
       anc_param_path (str): for example, "/path/to/your/anchor_param.npy

    Returns:
       np.array: num_anchor x 5. 5 stands for the feature of each anchor: [\theta, \phi, d_q, 1/0, d].
       \theta, \phi is the spherical coordinate of the query point from the anchor;
       d_q is the distance between query point and the anchor;
       1/0 is whether the query point is within the  anchor's mask;
       d is the distance on the SH sphere, which is the distance between the anchor and the surface along the line of the query point and the anchor, d=0 if the query point is not within the mask.
    """
    mash = Mash().fromParamsFile(anc_param_path)

    anc_pos = np.load(anc_param_path, allow_pickle=True).item()["params"][:, :3]

    ftrs = []
    for i in range(asdf_model.anchorNum()):
        print(f"processing {i}'th anchor...")
        global_directions = Directions.from_numpy(anc_pos[i, :] - qry)  # n_qry x 1
        theta_and_phi = global_directions.toPolars().numpy()  # n_qry x 2

        dq = np.linalg.norm(anc_pos[i, :] - qry, axis=1)  # n_qry x 1
        dq = dq.reshape(-1, 1)

        # global_directions  = toData(global_directions, 'numpy')
        global_directions = global_directions.numpy()
        in_anchor_i_mask_bool = asdf_model.getAnchorInMaskDirectionMask(
            i, global_directions
        )
        in_anchor_i_mask_bool = toData(in_anchor_i_mask_bool, "numpy")  # n_qry x 1 ?

        is_in_mask_float64 = in_anchor_i_mask_bool.astype(np.float64)

        # d_list = np.zeros_like(is_in_mask_float64, dtype=np.float64)
        global_directions = toData(global_directions, "torch")
        d_list = asdf_model.getAnchorDetectSHDists(i, global_directions)
        d_list = toData(d_list, "numpy")

        for i_qry, is_true in np.ndenumerate(in_anchor_i_mask_bool):
            if not is_true:  # FIXME
                d_list[i_qry] = 0

        is_in_mask_float64 = (
            np.asarray(is_in_mask_float64).reshape(-1, 1).astype(np.float64)
        )  # n_qry x 1
        d_list = np.array(d_list).reshape(-1, 1)  # n_qry x 1

        ftr = np.hstack((theta_and_phi, dq, is_in_mask_float64, d_list))
        ftrs.append(ftr)

    ftrs = np.asarray(ftrs)
    print(ftrs.shape)
    ftrs = ftrs.transpose(1, 0, 2)

    return np.asarray(ftrs)
