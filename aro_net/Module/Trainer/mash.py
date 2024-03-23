import os
import time
import glob
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


from aro_net.Dataset.mash import MashDataset
from aro_net.Method.time import getCurrentTime
from aro_net.Model.mash import MashNet

from aro_net.Module.logger import Logger


def cal_acc(x, gt, pred_type):
    if pred_type == "occ":
        acc = ((x["occ_pred"].sigmoid() > 0.5) == (gt["occ"] > 0.5)).float().sum(
            dim=-1
        ) / x["occ_pred"].shape[1]
    else:
        acc = ((x["sdf_pred"] >= 0) == (gt["sdf"] >= 0)).float().sum(dim=-1) / x[
            "sdf_pred"
        ].shape[1]
    acc = acc.mean(-1)
    return acc


def cal_loss_pred(x, gt, pred_type):
    if pred_type == "occ":
        loss_pred = F.binary_cross_entropy_with_logits(x["occ_pred"], gt["occ"])
    else:
        loss_pred = F.l1_loss(x["sdf_pred"], gt["sdf"])

    # print(f"pred: {len(x['occ_pred']<=0)}/{len(x['occ_pred'])}")

    return loss_pred


class Trainer(object):
    def __init__(self, args) -> None:
        self.args = args

        self.device = "cuda"

        current_time = getCurrentTime()

        self.dir_ckpt = "./output/" + current_time + "/"
        self.log_folder_path = "./logs/" + current_time + "/"

        self.train_loader = DataLoader(
            MashDataset(split="train", args=self.args),
            shuffle=True,
            batch_size=self.args.n_bs,
            num_workers=self.args.n_wk,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            MashDataset(split="val", args=self.args),
            shuffle=False,
            batch_size=self.args.n_bs,
            num_workers=self.args.n_wk,
            drop_last=True,
        )

        n_anc = 40  # FIXME: Remark! suppose we use 40 anchors!
        self.model = MashNet(
            n_anc=n_anc,
            n_qry=args.n_qry,
            n_local=args.n_local,
            cone_angle_th=args.cone_angle_th,
            tfm_pos_enc=args.tfm_pos_enc,
            cond_pn=args.cond_pn,
            use_dist_hit=args.use_dist_hit,
            pn_use_bn=args.pn_use_bn,
            pred_type=args.pred_type,
            norm_coord=args.norm_coord,
        ).to(self.device)

        if self.args.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        self.writer = Logger(self.log_folder_path)
        return

    def train_step(self, batch, opt):
        for key in batch:
            batch[key] = batch[key].cuda()
        opt.zero_grad()
        x = self.model(batch)

        loss_pred = cal_loss_pred(x, batch, self.args.pred_type)
        loss = loss_pred
        if self.args.use_dist_hit:
            loss_hit_dist = F.l1_loss(x["dist_hit_pred"], batch["dist_hit"])
            loss += loss_hit_dist
        else:
            loss_hit_dist = torch.zeros(1)

        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = cal_acc(x, batch, self.args.pred_type)
        return loss_pred.item(), loss_hit_dist.item(), acc.item()

    @torch.no_grad()
    def val_step(self):
        avg_loss_pred = 0
        avg_acc = 0
        ni = 0
        for batch in self.val_loader:
            for key in batch:
                batch[key] = batch[key].cuda()
            x = self.model(batch)

            loss_pred = cal_loss_pred(x, batch, self.args.pred_type)

            acc = cal_acc(x, batch, self.args.red_type)

            avg_loss_pred = avg_loss_pred + loss_pred.item()
            avg_acc = avg_acc + acc.item()
            ni += 1
        avg_loss_pred /= ni
        avg_acc /= ni
        return avg_loss_pred, avg_acc

    def train(self, args):
        os.makedirs(self.dir_ckpt, exist_ok=True)

        opt = optim.Adam(self.model.parameters(), lr=args.lr)

        if args.resume:
            fnames_ckpt = glob.glob(os.path.join(self.dir_ckpt, "*"))
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

        for i in range(epoch_latest, args.n_epochs):
            start_time = time.process_time()
            self.model.train()
            for batch in tqdm(self.train_loader):
                loss_pred, loss_hit_dist, acc = self.train_step(batch, opt)
                if n_iter % args.freq_log == 0:
                    print(
                        "[train] epcho:",
                        n_epoch,
                        " ,iter:",
                        n_iter,
                        " loss_pred:",
                        loss_pred,
                        " loss_hit_dist:",
                        loss_hit_dist,
                        " acc:",
                        acc,
                    )
                    self.writer.addScalar("Loss/train", loss_pred, n_iter)
                    self.writer.addScalar("Acc/train", acc, n_iter)
                    lr = opt.state_dict()["param_groups"][0]["lr"]
                    self.writer.addScalar("Acc/lr", lr, n_iter)

                n_iter += 1
            end_time = time.process_time()
            execution_time = end_time - start_time
            print(f"epoch {i} finished, costing {execution_time/60.0} minutes")

            if n_epoch % args.freq_ckpt == 0:
                # model.eval() # avg_loss_pred, avg_acc = val_step(model, val_loader, args.pred_type)
                # writer.addScalar('Loss/val', avg_loss_pred, n_iter)
                # writer.addScalar('Acc/val', avg_acc, n_iter)
                # print('[val] epcho:', n_epoch,' ,iter:',n_iter," avg_loss_pred:",avg_loss_pred, " acc:",avg_acc)
                if args.multi_gpu:
                    torch.save(
                        {
                            "model": self.model.module.state_dict(),
                            "opt": opt.state_dict(),
                            "n_epoch": n_epoch,
                            "n_iter": n_iter,
                        },
                        f"{self.dir_ckpt}/{n_epoch}_{n_iter}_{start_time}.ckpt",
                    )
                # {avg_loss_pred:.4}_{avg_acc:.4}.ckpt')
                else:
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "opt": opt.state_dict(),
                            "n_epoch": n_epoch,
                            "n_iter": n_iter,
                        },
                        f"{self.dir_ckpt}/{n_epoch}_{n_iter}_{start_time}.ckpt",
                    )  # {avg_loss_pred:.4}_{avg_acc:.4}.ckpt')
            if n_epoch > 0 and n_epoch % args.freq_decay == 0:
                for g in opt.param_groups:
                    g["lr"] = g["lr"] * args.weight_decay

            n_epoch += 1

        return True
