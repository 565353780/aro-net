import sys

sys.path.append("../ma-sh")

from aro_net.Dataset.aro import ARONetDataset
from aro_net.Dataset.mash import MashDataset


def test():
    aro_train_dataset = ARONetDataset("train")
    mash_train_dataset = MashDataset("train")

    for i, aro_data in enumerate(aro_train_dataset):
        print(aro_data["occ"])
        print(aro_data.keys())
        break

    for i, mash_data in enumerate(mash_train_dataset):
        print(mash_data["occ"])
        print(mash_data.keys())
        break

    for key, item in aro_train_dataset[0].items():
        print("aro:", key, item.shape)
    for key, item in mash_train_dataset[0].items():
        print("mash:", key, item.shape)
    return True
