import os
import numpy as np
from tqdm import tqdm
from math import ceil
from typing import Tuple

from aro_net.Method.sample import sampleQueryPoints
from aro_net.Method.feature import toMashFileAnchorFeature


class Convertor(object):
    def __init__(self, dataset_root_folder_path: str) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        return

    def convertToSplits(
        self,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> Tuple[list, list, list]:
        if not os.path.exists(self.dataset_root_folder_path):
            print("[ERROR][Convertor::convertToSplits]")
            print("\t dataset root folder not exist!")
            print("\t dataset_root_folder_path:", self.dataset_root_folder_path)
            return [], [], []

        filename_list = os.listdir(self.dataset_root_folder_path)

        file_basename_list = [filename[:-8] for filename in filename_list]

        permut_file_basename_list = np.random.permutation(file_basename_list)

        file_basename_num = len(file_basename_list)

        train_split_num = ceil(train_scale * file_basename_num)
        val_split_num = ceil(val_scale * file_basename_num)

        train_split = permut_file_basename_list[:train_split_num]
        val_split = permut_file_basename_list[
            train_split_num : train_split_num + val_split_num
        ]
        test_split = permut_file_basename_list[train_split_num + val_split_num :]

        return train_split.tolist(), val_split.tolist(), test_split.tolist()

    def convertToSplitFiles(
        self,
        save_split_folder_path: str,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> bool:
        train_split, val_split, test_split = self.convertToSplits(
            train_scale, val_scale
        )

        if len(train_split) + len(val_split) + len(test_split) == 0:
            print("[ERROR][Convertor::convertToSplitFiles]")
            print("\t convertToSplits failed!")
            return False

        os.makedirs(save_split_folder_path, exist_ok=True)

        with open(save_split_folder_path + "train.lst", "w") as f:
            for train_name in train_split:
                f.write(train_name + "\n")

        with open(save_split_folder_path + "val.lst", "w") as f:
            for val_name in val_split:
                f.write(val_name + "\n")

        with open(save_split_folder_path + "test.lst", "w") as f:
            for test_name in test_split:
                f.write(test_name + "\n")
        return True

    def convertToAnchorFeatures(
        self, query_point_root_folder_path: str, save_feature_folder_path: str
    ) -> bool:
        if not os.path.exists(self.dataset_root_folder_path):
            print("[ERROR][Convertor::convertToSplits]")
            print("\t dataset root folder not exist!")
            print("\t dataset_root_folder_path:", self.dataset_root_folder_path)
            return False

        if not os.path.exists(query_point_root_folder_path):
            print("[ERROR][Convertor::convertToSplits]")
            print("\t query point root folder not exist!")
            print("\t query_point_root_folder_path:", query_point_root_folder_path)
            return False

        os.makedirs(save_feature_folder_path, exist_ok=True)

        filename_list = os.listdir(self.dataset_root_folder_path)

        for filename in tqdm(filename_list):
            save_feature_file_path = save_feature_folder_path + filename

            if os.path.exists(save_feature_file_path):
                continue

            file_path = self.dataset_root_folder_path + filename

            shape_file_name = filename.replace("_obj", "").replace("_ply", "")

            query_point_file_path = query_point_root_folder_path + shape_file_name

            if not os.path.exists(query_point_file_path):
                print("[WARN][Convertor::convertToAnchorFeatures]")
                print("\t query point file not exist!")
                print("\t query_point_file_path:", query_point_file_path)
                continue

            query_points = np.load(query_point_file_path)

            anchor_feature = (
                toMashFileAnchorFeature(file_path, query_points).cpu().numpy()
            )

            np.save(save_feature_file_path, anchor_feature)

        return True
