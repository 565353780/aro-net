import sys

sys.path.append("../ma-sh")

from aro_net.Module.convertor import Convertor


def demo():
    dataset_root_folder_path = (
        "/home/chli/Dataset/aro_net/data/shapenet/mash/100anc/mash/02691156/"
    )
    save_split_folder_path = (
        "/home/chli/Dataset/aro_net/data/shapenet/04_splits/02691156/mash/"
    )
    save_feature_folder_path = (
        "/home/chli/Dataset/aro_net/data/shapenet/anchor_feature/100anc/02691156/"
    )
    train_scale = 0.8
    val_scale = 0.1

    convertor = Convertor(dataset_root_folder_path)
    convertor.convertToSplitFiles(save_split_folder_path, train_scale, val_scale)
    convertor.convertToAnchorFeatures(save_feature_folder_path)
    return True
