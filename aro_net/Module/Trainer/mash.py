import os
import glob
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


from aro_net.Config.config import MASH_CONFIG
from aro_net.Dataset.mash import MashDataset
from aro_net.Method.time import getCurrentTime
from aro_net.Model.mash import MashNet

from aro_net.Module.logger import Logger


def cal_acc(x, gt):
    acc = ((x.sigmoid() > 0.5) == (gt["occ"] > 0.5)).float().sum(dim=-1) / x.shape[1]
    acc = acc.mean(-1)
    return acc


def cal_loss_pred(x, gt):
    loss_pred = F.binary_cross_entropy_with_logits(x, gt["occ"])
    return loss_pred


class Trainer(object):
    def __init__(self) -> None:
        current_time = getCurrentTime()

        self.dir_ckpt = "./output/" + current_time + "/"
        self.log_folder_path = "./logs/" + current_time + "/"

        self.train_loader = DataLoader(
            MashDataset("train"),
            shuffle=True,
            batch_size=MASH_CONFIG.n_bs,
            num_workers=MASH_CONFIG.n_wk,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            MashDataset("val"),
            shuffle=False,
            batch_size=MASH_CONFIG.n_bs,
            num_workers=MASH_CONFIG.n_wk,
            drop_last=True,
        )

        self.model = MashNet().to(MASH_CONFIG.device)

        self.writer = Logger(self.log_folder_path)
        return

    def train_step(self, batch, opt):
        for key in batch:
            batch[key] = batch[key].to(MASH_CONFIG.device)
        opt.zero_grad()
        x = self.model(batch)

        loss = cal_loss_pred(x, batch)

        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = cal_acc(x, batch)
        return loss.item(), acc.item()

    @torch.no_grad()
    def val_step(self):
        avg_loss_pred = 0
        avg_acc = 0
        ni = 0

        print("[INFO][Trainer::val_step]")
        print("\t start val loss and acc...")
        for batch in tqdm(self.val_loader):
            for key in batch:
                try:
                    batch[key] = batch[key].to(MASH_CONFIG.device)
                except:
                    pass
            x = self.model(batch)

            loss_pred = cal_loss_pred(x, batch)

            acc = cal_acc(x, batch)

            avg_loss_pred = avg_loss_pred + loss_pred.item()
            avg_acc = avg_acc + acc.item()
            ni += 1

        avg_loss_pred /= ni
        avg_acc /= ni
        return avg_loss_pred, avg_acc

    def train(self):
        os.makedirs(self.dir_ckpt, exist_ok=True)

        opt = optim.Adam(self.model.parameters(), lr=MASH_CONFIG.lr)

        fnames_ckpt = glob.glob(os.path.join(self.dir_ckpt, "*"))
        if len(fnames_ckpt) > 0:
            fname_ckpt_latest = max(fnames_ckpt, key=os.path.getctime)
            # path_ckpt = os.path.join(dir_ckpt, fname_ckpt_latest)
            ckpt = torch.load(fname_ckpt_latest)
            self.model.module.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            epoch_latest = ckpt["n_epoch"] + 1
            n_iter = ckpt["n_iter"]
            n_epoch = epoch_latest
        else:
            epoch_latest = 0
            n_iter = 0
            n_epoch = 0

        for i in range(epoch_latest, MASH_CONFIG.n_epochs):
            self.model.train()
            print("[INFO][Trainer::train]")
            print("\t start train mash occ itr", i + 1, "...")
            for batch in tqdm(self.train_loader):
                loss, acc = self.train_step(batch, opt)
                lr = opt.state_dict()["param_groups"][0]["lr"]

                self.writer.addScalar("Loss/train", loss, n_iter)
                self.writer.addScalar("Acc/train", acc, n_iter)
                self.writer.addScalar("Acc/lr", lr, n_iter)

                n_iter += 1

            self.model.eval()
            avg_loss_pred, avg_acc = self.val_step()
            self.writer.addScalar("Loss/val", avg_loss_pred, n_iter)
            self.writer.addScalar("Acc/val", avg_acc, n_iter)
            print(
                "[val] epcho:",
                n_epoch,
                " ,iter:",
                n_iter,
                " avg_loss_pred:",
                avg_loss_pred,
                " acc:",
                avg_acc,
            )

            torch.save(
                {
                    "model": self.model.state_dict(),
                    "opt": opt.state_dict(),
                    "n_epoch": n_epoch,
                    "n_iter": n_iter,
                },
                f"{self.dir_ckpt}/{n_epoch}_{n_iter}.ckpt",
            )

            if n_epoch > 0 and n_epoch % MASH_CONFIG.freq_decay == 0:
                for g in opt.param_groups:
                    g["lr"] = g["lr"] * MASH_CONFIG.weight_decay

            n_epoch += 1

        return True
