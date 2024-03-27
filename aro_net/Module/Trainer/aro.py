import os
import glob
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


from aro_net.Config.config import ARO_CONFIG
from aro_net.Dataset.aro import ARONetDataset
from aro_net.Dataset.single_shape import SingleShapeDataset
from aro_net.Model.aro import ARONet

from aro_net.Method.time import getCurrentTime

from aro_net.Module.logger import Logger


def cal_acc(x, gt):
    acc = ((x["occ_pred"].sigmoid() > 0.5) == (gt["occ"] > 0.5)).float().sum(
        dim=-1
    ) / x["occ_pred"].shape[1]
    acc = acc.mean(-1)
    return acc


def cal_loss_pred(x, gt):
    loss_pred = F.binary_cross_entropy_with_logits(x["occ_pred"], gt["occ"])
    return loss_pred


class Trainer(object):
    def __init__(self) -> None:
        current_time = getCurrentTime()

        self.dir_ckpt = "./output/" + current_time + "/"
        self.log_folder_path = "./logs/" + current_time + "/"

        self.train_loader = DataLoader(
            ARONetDataset("train"),
            shuffle=True,
            batch_size=ARO_CONFIG.n_bs,
            num_workers=ARO_CONFIG.n_wk,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            ARONetDataset("val"),
            shuffle=False,
            batch_size=ARO_CONFIG.n_bs,
            num_workers=ARO_CONFIG.n_wk,
            drop_last=True,
        )

        self.model = ARONet().to(ARO_CONFIG.device)

        if ARO_CONFIG.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        self.writer = Logger(self.log_folder_path)
        return

    def train_step(self, batch, opt):
        for key in batch:
            batch[key] = batch[key].to(ARO_CONFIG.device)
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
        for batch in self.val_loader:
            for key in batch:
                batch[key] = batch[key].to(ARO_CONFIG.device)
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

        opt = optim.Adam(self.model.parameters(), lr=ARO_CONFIG.lr)

        fnames_ckpt = glob.glob(os.path.join(self.dir_ckpt, "*"))
        if len(fnames_ckpt) > 0:
            fname_ckpt_latest = max(fnames_ckpt, key=os.path.getctime)
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

        for _ in range(epoch_latest, ARO_CONFIG.n_epochs):
            self.model.train()
            for batch in tqdm(self.train_loader):
                loss, acc = self.train_step(batch, opt)
                if n_iter % ARO_CONFIG.freq_log == 0:
                    print(
                        "[train] epcho:",
                        n_epoch,
                        " ,iter:",
                        n_iter,
                        " loss:",
                        loss,
                        " acc:",
                        acc,
                    )
                    self.writer.addScalar("Loss/train", loss, n_iter)
                    self.writer.addScalar("Acc/train", acc, n_iter)

                n_iter += 1

            if n_epoch % ARO_CONFIG.freq_ckpt == 0:
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
                if ARO_CONFIG.multi_gpu:
                    torch.save(
                        {
                            "model": self.model.module.state_dict(),
                            "opt": opt.state_dict(),
                            "n_epoch": n_epoch,
                            "n_iter": n_iter,
                        },
                        f"{self.dir_ckpt}/{n_epoch}_{n_iter}_{avg_loss_pred:.4}_{avg_acc:.4}.ckpt",
                    )
                else:
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "opt": opt.state_dict(),
                            "n_epoch": n_epoch,
                            "n_iter": n_iter,
                        },
                        f"{self.dir_ckpt}/{n_epoch}_{n_iter}_{avg_loss_pred:.4}_{avg_acc:.4}.ckpt",
                    )
            if n_epoch > 0 and n_epoch % ARO_CONFIG.freq_decay == 0:
                for g in opt.param_groups:
                    g["lr"] = g["lr"] * ARO_CONFIG.weight_decay

            n_epoch += 1

        return True
