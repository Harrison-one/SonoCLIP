import os
import sys
import subprocess
import argparse
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAIN_ROOT)
for path in (TRAIN_ROOT, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import sonoclip
import loralib as lora

from utils import concat_all_gather, is_dist_avail_and_initialized
from dataset.ul_s_test import UL_Test
from dataset.sono_ul_txt import Sono_UL_Txt
from scheduler import cosine_lr

simple_templates = [
    "A fetal ultrasound {} view."
]


def get_rank_safe():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size_safe():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, disable=(get_rank_safe() != 0)):
            texts = [template.format(classname) for template in templates]
            texts = sonoclip.tokenize(texts).cuda(non_blocking=True)
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda(non_blocking=True)
    return zeroshot_weights


class CLIP_Clean_Train:
    def __init__(
        self,
        local_rank=0,
        lr=4e-5,
        weigth_decay=0.02,
        log_scale=4.6052,
        lora_rank=-1,
        common_pair=0.0,
        para_gamma=0.01,
        alpha_in_channels: int = 1,
        exp_name="auto",
        warmup_length=200,
        epoch_num=1,
        subnum=10000,
        use_clip_loss=False,
        base_model_path=None,
    ):
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)
        self.alpha_in_channels = alpha_in_channels

        if lora_rank == -1:
            if not base_model_path:
                raise ValueError("--base_model_path is required when lora_rank=-1")
            self.model, _ = sonoclip.load(
                base_model_path,
                device="cpu",
                lora_adapt=False,
                rank=-1,
                alpha_in_channels=alpha_in_channels,
                
            )
        else:
            self.model, _ = sonoclip.load(
                "ViT-L/14",
                device="cpu",
                lora_adapt=True,
                rank=lora_rank,
                alpha_in_channels=alpha_in_channels,
                
            )

        self.model = self.model.float().to(self.device)
        self.batch_size = 16 // 2
        self.num_epoch = int(1e10) if epoch_num is None else int(epoch_num)
        self.lr = lr
        self.subnum = subnum
        self.para_gamma = para_gamma
        self.use_clip_loss = use_clip_loss  # True: 使用原始CLIP loss, False: 使用SigLIP loss

        if exp_name == "auto":
            self.logdir = (
                f"log/grit_1m/lr={lr}_wd={weigth_decay}_wl={warmup_length}"
                f"_logs={log_scale}_L14_336_lora={lora_rank}_cp={common_pair}"
                f"_para_gamma={para_gamma}_e{self.num_epoch}_16xb_subnum={self.subnum}"
            )
        else:
            self.logdir = exp_name

        self.ckptdir = os.path.join(self.logdir, "ckpt")
        os.makedirs(self.ckptdir, exist_ok=True)

        self.test_results_txt = os.path.join(self.logdir, "test_results.txt")
        self.writer = SummaryWriter(self.logdir)

        # DDP only wraps visual branch.
        # visual forward path is fixed, so no need find_unused_parameters=True.
        self.model.visual = torch.nn.parallel.DistributedDataParallel(
            self.model.visual,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

        # SigLIP params: t = exp(t0), logits = sim * t + b
        # 原始CLIP: 只需要 logit_scale，不需要 bias
        if use_clip_loss:
            # 原始CLIP: 使用 log_scale 作为初始值
            self.model.logit_scale = torch.nn.Parameter(
                torch.tensor(log_scale, dtype=torch.float32, device=self.device)
            )
        else:
            # SigLIP: 需要 logit_scale 和 logit_bias
            t0_init = np.log(10.0)
            self.model.logit_scale = torch.nn.Parameter(
                torch.tensor(t0_init, dtype=torch.float32, device=self.device)
            )
            self.model.logit_bias = torch.nn.Parameter(
                torch.tensor(-10.0, dtype=torch.float32, device=self.device)
            )

        conv_opt_paras = []
        other_opt_paras = []
        if use_clip_loss:
            logit_paras = [self.model.logit_scale]  # 原始CLIP只需要logit_scale
        else:
            logit_paras = [self.model.logit_scale, self.model.logit_bias]  # SigLIP需要两者

        if lora_rank != -1:
            lora.mark_only_lora_as_trainable(self.model)
            for k, v in self.model.named_parameters():
                if "logit_scale" in k or "logit_bias" in k:
                    continue
                if v.requires_grad:
                    other_opt_paras.append(v)
                elif "conv1_alpha" in k:
                    v.requires_grad_(True)
                    conv_opt_paras.append(v)

            self.model.logit_scale.requires_grad_(True)
            if not use_clip_loss:
                self.model.logit_bias.requires_grad_(True)

        else:
            for _, v in self.model.named_parameters():
                v.requires_grad_(False)

            for k, v in self.model.visual.named_parameters():
                v.requires_grad_(True)
                if "conv1_alpha" in k:
                    conv_opt_paras.append(v)
                else:
                    other_opt_paras.append(v)

            self.model.logit_scale.requires_grad_(True)
            if not use_clip_loss:
                self.model.logit_bias.requires_grad_(True)

        self.optimizer = optim.AdamW(
            [
                {"params": conv_opt_paras, "lr": self.lr, "weight_decay": weigth_decay},
                {"params": other_opt_paras, "lr": self.lr * para_gamma, "weight_decay": weigth_decay},
                {"params": logit_paras, "lr": self.lr * para_gamma, "weight_decay": weigth_decay},
            ]
        )

        self.scaler = torch.cuda.amp.GradScaler()

    def _sync_non_ddp_grads(self):
        """Sync grads for trainable params not covered by DDP."""
        if not is_dist_avail_and_initialized():
            return
        params_to_sync = [self.model.logit_scale]
        if not self.use_clip_loss:
            params_to_sync.append(self.model.logit_bias)
        for p in params_to_sync:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= get_world_size_safe()

    def _save_checkpoint(self, step, amp=False):
        if get_rank_safe() != 0:
            return

        ckpt = {
            "visual": self.model.visual.state_dict(),
            "logit_scale": self.model.logit_scale.detach().float().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
        }
        if not self.use_clip_loss:
            ckpt["logit_bias"] = self.model.logit_bias.detach().float().cpu()
        if amp:
            ckpt["scaler"] = self.scaler.state_dict()

        save_path = os.path.join(self.ckptdir, f"iter_{step}.pth")
        torch.save(ckpt, save_path)

    def _load_checkpoint(self, resume_path, amp=False):
        map_location = {"cuda:0": f"cuda:{self.local_rank}"}
        ckpt = torch.load(resume_path, map_location=map_location)

        # backward compatibility
        if "visual" in ckpt:
            self.model.visual.load_state_dict(ckpt["visual"], strict=False)
            self.model.logit_scale.data.copy_(ckpt["logit_scale"].to(self.device))
            if "logit_bias" in ckpt and not self.use_clip_loss:
                self.model.logit_bias.data.copy_(ckpt["logit_bias"].to(self.device))

            if "optimizer" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                    # move optimizer states to current device
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(self.device)
                except ValueError as exc:
                    if get_rank_safe() == 0:
                        print(f"skip optimizer state: {exc}")

            if amp and "scaler" in ckpt:
                try:
                    self.scaler.load_state_dict(ckpt["scaler"])
                except RuntimeError as exc:
                    if get_rank_safe() == 0:
                        print(f"skip scaler state: {exc}")

            resume_iter = int(ckpt.get("step", 0))
        else:
            # old format: only visual state_dict
            self.model.visual.load_state_dict(ckpt, strict=False)
            resume_iter = int(os.path.basename(resume_path)[5:-4])

        if get_rank_safe() == 0:
            print(f"load resumed checkpoint: {os.path.basename(resume_path)}")
        return resume_iter

    def inference(self, images, masks, texts):
        image_features = self.model.visual(images, masks)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # concat_all_gather is no_grad, so compute rank-local partial losses
        image_feat_all = concat_all_gather(image_features)
        text_feat_all = concat_all_gather(text_features)

        B = image_features.size(0)
        dev = image_features.device
        rank = get_rank_safe()

        if self.use_clip_loss:
            # 原始CLIP loss: 使用交叉熵 + label smoothing
            sim_i2t = torch.matmul(image_features, text_feat_all.T)
            sim_t2i = torch.matmul(image_feat_all, text_features.T)
            sim_t2i = sim_t2i.T

            logit_scale = self.model.logit_scale.exp()
            sim_i2t = logit_scale * sim_i2t
            sim_t2i = logit_scale * sim_t2i

            targets = torch.linspace(rank * B, rank * B + B - 1, B, dtype=torch.long).to(dev)
            loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2.0
            return loss_itc
        else:
            # SigLIP loss: 使用 logsigmoid
            n = image_feat_all.shape[0]
            t = self.model.logit_scale.exp()
            b = self.model.logit_bias

            labels_full = 2.0 * torch.eye(n, device=dev, dtype=image_features.dtype) - 1.0

            # local image -> all text (grad flows to local image_features)
            logits_i2t = (image_features @ text_feat_all.T) * t + b
            labels_our_rows = labels_full[rank * B: (rank + 1) * B, :]
            loss_i2t_r = -F.logsigmoid(labels_our_rows * logits_i2t).sum() / n

            # all image -> local text (grad flows to local text_features)
            logits_t2i = (image_feat_all @ text_features.T) * t + b
            labels_our_cols = labels_full[:, rank * B: (rank + 1) * B]
            loss_t2i_r = -F.logsigmoid(labels_our_cols * logits_t2i).sum() / n

            loss_r = (loss_i2t_r + loss_t2i_r) / 2.0
            return loss_r

    def train_epoch(self, dataloader, test_loaders, epoch, start_iter=0, amp=False):
        running_loss = 0.0
        batch_num = 0
        num_batches_per_epoch = len(dataloader)

        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        for i, (images, masks, texts) in enumerate(tqdm(dataloader, disable=(get_rank_safe() != 0))):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue

            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler(step)

            images = images.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            texts = sonoclip.tokenize(texts).cuda(non_blocking=True)

            if amp:
                with torch.cuda.amp.autocast():
                    loss = self.inference(images, masks, texts)

                self.scaler.scale(loss).backward()
                self._sync_non_ddp_grads()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.inference(images, masks, texts)
                loss.backward()
                self._sync_non_ddp_grads()
                self.optimizer.step()

            running_loss += loss.item()
            batch_num += 1

            if step > 0 and step % 5000 == 0:
                loss_for_log = torch.tensor(loss.item(), device=self.device)
                if is_dist_avail_and_initialized():
                    dist.all_reduce(loss_for_log, op=dist.ReduceOp.SUM)
                    loss_for_log /= get_world_size_safe()
                loss_for_log = loss_for_log.item()

                if get_rank_safe() == 0:
                    self.writer.add_scalar("hyper/lr", self.optimizer.param_groups[0]["lr"], step)
                    self.writer.add_scalar("logit_scale/train", self.model.logit_scale.item(), step)
                    if not self.use_clip_loss:
                        self.writer.add_scalar("logit_bias/train", self.model.logit_bias.item(), step)

                    print("=====================================")
                    print(f"train lr (alpha conv) step {step}: {self.optimizer.param_groups[0]['lr']}")
                    print(f"train lr (other layer) step {step}: {self.optimizer.param_groups[1]['lr']}")
                    if self.use_clip_loss:
                        print(
                            f"train logit_scale step {step}: {self.model.logit_scale.item()}, "
                            f"temperature={self.model.logit_scale.exp().item():.4f}"
                        )
                    else:
                        print(
                            f"train logit_scale (t_prime) step {step}: {self.model.logit_scale.item()}, "
                            f"t={self.model.logit_scale.exp().item():.4f}, "
                            f"bias={self.model.logit_bias.item():.4f}"
                        )
                    print(f"train loss step {step}: {loss_for_log}")
                    print("=====================================")

                    self.writer.add_scalar("Loss/train", loss_for_log, step)

                    if step % 5000 == 0 and step != 0 and step > 800:
                        self._save_checkpoint(step, amp=amp)

                with torch.no_grad():
                    self.model.visual.eval()
                    for test_name, test_loader in test_loaders.items():
                        self.text_embeddings = zeroshot_classifier(
                            test_loader.dataset.classes, simple_templates, self.model
                        )
                        temp_corr_dict = self.test_epoch(test_loader)

                        if is_dist_avail_and_initialized():
                            output = [None] * get_world_size_safe()
                            dist.all_gather_object(output, temp_corr_dict)
                        else:
                            output = [temp_corr_dict]

                        if get_rank_safe() == 0:
                            final_dict = {}
                            for dic in output:
                                for k, v in dic.items():
                                    if k not in final_dict:
                                        final_dict[k] = v
                                    else:
                                        final_dict[k][0] += v[0]
                                        final_dict[k][1] += v[1]
                                        final_dict[k][2] += v[2]

                            acc1 = 0.0
                            acc5 = 0.0
                            num_class = 0
                            for v in final_dict.values():
                                acc1 += v[1] / v[0]
                                acc5 += v[2] / v[0]
                                num_class += 1
                            acc1 /= num_class
                            acc5 /= num_class

                            print("=====================================")
                            print(f"test {test_name} acc-1 step {step}: {acc1}")
                            print(f"test {test_name} acc-5 step {step}: {acc5}")
                            print("=====================================")

                            self.writer.add_scalar(f"{test_name}_Acc1/test", acc1, step)
                            self.writer.add_scalar(f"{test_name}_Acc5/test", acc5, step)
                            self._append_test_result(step, test_name, acc1, acc5)

                    self.model.visual.train()

        return running_loss / max(batch_num, 1)

    @torch.no_grad()
    def test_epoch(self, dataloader):
        temp_corr_dict = {}
        for images, masks, target in tqdm(dataloader, disable=(get_rank_safe() != 0)):
            images = images.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            image_features = self.model.visual(images, masks)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            score = torch.matmul(image_features, self.text_embeddings)
            pred = score.topk(1, dim=1)[1].squeeze(dim=1)
            pred_5 = score.topk(5, dim=1)[1]

            for idx in range(target.shape[0]):
                cls = target[idx].item()
                if cls not in temp_corr_dict:
                    temp_corr_dict[cls] = [0, 0, 0]
                temp_corr_dict[cls][0] += 1
                if cls == pred[idx].item():
                    temp_corr_dict[cls][1] += 1
                if cls in pred_5[idx].tolist():
                    temp_corr_dict[cls][2] += 1
        return temp_corr_dict

    def _append_test_result(self, step, test_name, acc1, acc5):
        write_header = (
            (not os.path.isfile(self.test_results_txt)) or (os.path.getsize(self.test_results_txt) == 0)
        )
        with open(self.test_results_txt, "a", encoding="utf-8") as f:
            if write_header:
                f.write("step\ttest_name\tacc1\tacc5\n")
            f.write(f"{step}\t{test_name}\t{acc1:.6f}\t{acc5:.6f}\n")
            f.flush()

    def test(self, epoch=0, data_root=None):
        self.model.visual.eval()
        if data_root is None:
            raise ValueError("data_root is required for test()")
        data_root = os.path.abspath(data_root)
        testset = UL_Test(
            ann_files=os.path.join(data_root, "test", "ul_test.json"),
            image_root=os.path.join(data_root, "test", "images"),
            plane_ids_txt=os.path.join(data_root, "test", "plane_ids.txt"),
        )
        self.text_embeddings = zeroshot_classifier(testset.classes, simple_templates, self.model)

        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=16,
            pin_memory=True,
        )

        with torch.no_grad():
            temp_corr_dict = self.test_epoch(testloader)

            if is_dist_avail_and_initialized():
                output = [None] * get_world_size_safe()
                dist.all_gather_object(output, temp_corr_dict)
            else:
                output = [temp_corr_dict]

            if self.local_rank == 0:
                final_dict = {}
                for dic in output:
                    for k, v in dic.items():
                        if k not in final_dict:
                            final_dict[k] = v
                        else:
                            final_dict[k][0] += v[0]
                            final_dict[k][1] += v[1]
                            final_dict[k][2] += v[2]

                acc1 = 0.0
                acc5 = 0.0
                num_class = 0
                for v in final_dict.values():
                    acc1 += v[1] / v[0]
                    acc5 += v[2] / v[0]
                    num_class += 1
                acc1 /= num_class
                acc5 /= num_class

                print("=====================================")
                print(f"test mean of per class acc-1 step 0: {acc1}")
                print(f"test mean of per class acc-5 step 0: {acc5}")
                print("=====================================")

                self._append_test_result(0, "Ul_Test", acc1, acc5)

    def train(
        self,
        common_pair=False,
        resume=False,
        amp=False,
        warmup_length=200,
        resume_path=None,
        data_root=None,
    ):
        if data_root is None:
            raise ValueError("data_root is required for train()")
        data_root = os.path.abspath(data_root)
        test_ann_file = os.path.join(data_root, "test", "ul_test.json")
        test_image_root = os.path.join(data_root, "test", "images")
        test_plane_ids_txt = os.path.join(data_root, "test", "plane_ids.txt")

        testset_ul_test = UL_Test(
            ann_files=test_ann_file,
            image_root=test_image_root,
            plane_ids_txt=test_plane_ids_txt,
            hi_res=True,
        )
        testset_ul_test_all_one = UL_Test(
            ann_files=test_ann_file,
            image_root=test_image_root,
            plane_ids_txt=test_plane_ids_txt,
            hi_res=True,
            all_one=True,
        )

        ids_file = os.path.join(data_root, "data", "train_stems.txt")
        json_root = os.path.join(data_root, "data", "jsons")
        image_root = os.path.join(data_root, "data", "images")

        if common_pair >= 1.0:
            try:
                from dataset.sono_ul_common_pair import SonoULCommonPair
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "common_pair>=1.0 requires dataset.sono_ul_common_pair, "
                    "but that module is not present in this checkout."
                ) from exc
            trainset = SonoULCommonPair(
                ids_file=ids_file,
                json_root=json_root,
                image_root=image_root,
                subnum=self.subnum,
                hi_res=True,
            )
        else:
            trainset = Sono_UL_Txt(
                ids_file=ids_file,
                root_pth=image_root,
                json_root=json_root,
                image_root=image_root,
                common_pair=common_pair,
                subnum=self.subnum,
                hi_res=True,
            )

        test_loaders = {}
        prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "1"))
        for name, testset in zip(
            ["Ul_Test", "Ul_Test_All_One"],
            [testset_ul_test, testset_ul_test_all_one],
        ):
            test_sampler = DistributedSampler(dataset=testset, shuffle=False)
            test_num_workers = int(os.environ.get("TEST_NUM_WORKERS", "2"))
            test_loader_kwargs = dict(
                dataset=testset,
                batch_size=self.batch_size,
                sampler=test_sampler,
                num_workers=test_num_workers,
                pin_memory=True,
            )
            if test_num_workers > 0:
                test_loader_kwargs.update(
                    persistent_workers=True,
                    prefetch_factor=prefetch_factor,
                )
            test_loader = torch.utils.data.DataLoader(**test_loader_kwargs)
            test_loaders[name] = test_loader

        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_num_workers = int(os.environ.get("TRAIN_NUM_WORKERS", "8"))
        train_loader_kwargs = dict(
            dataset=trainset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=train_num_workers,
            pin_memory=True,
        )
        if train_num_workers > 0:
            train_loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=prefetch_factor,
            )
        train_loader = torch.utils.data.DataLoader(**train_loader_kwargs)

        self.scheduler = cosine_lr(
            self.optimizer,
            base_lr=self.lr,
            warmup_length=warmup_length,
            steps=5000,
            para_gamma=self.para_gamma,
        )

        start_epoch = 0
        resume_iter = 0

        if resume:
            if resume_path is not None:
                resume_iter = self._load_checkpoint(resume_path, amp=amp)
                start_epoch = resume_iter // len(train_loader)
            elif len(os.listdir(self.ckptdir)) > 0:
                ckpt_files = sorted(
                    [f for f in os.listdir(self.ckptdir) if f.endswith(".pth")],
                    key=lambda x: int(x[5:-4]),
                )
                if len(ckpt_files) > 0:
                    resume_pth = ckpt_files[-1]
                    resume_iter = self._load_checkpoint(
                        os.path.join(self.ckptdir, resume_pth),
                        amp=amp,
                    )
                    start_epoch = resume_iter // len(train_loader)

        for epoch in range(start_epoch, self.num_epoch):
            if (len(trainset) * epoch) > 4000 * self.batch_size * 256:
                if get_rank_safe() == 0:
                    print(len(trainset), epoch)
                break

            self.train_epoch(
                train_loader,
                test_loaders,
                epoch,
                start_iter=resume_iter,
                amp=amp,
            )
            resume_iter = 0  # only skip on first resumed epoch


def setup_distributed(backend="nccl", port=None):
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")

        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29991"

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank % num_gpus


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="params")
    parser.add_argument("--lr", default=4e-5, type=float, help="lr.")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="wd.")
    parser.add_argument("--log_scale", default=4.6052, type=float, help="clip temperature log scale.")
    parser.add_argument("--lora_rank", default=-1, type=int, help="lora rank (-1 to not use lora).")
    parser.add_argument("--common_pair", default=0.0, type=float, help="propotion to use image with all 1 alpha and whole caption.")
    parser.add_argument("--para_gamma", default=0.01, type=float, help="para_gamma of other parameters")
    parser.add_argument("--use_clip_loss", action="store_true", help="Use original CLIP loss (cross_entropy) instead of SigLIP loss (logsigmoid)")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--resume_path", default=None, type=str, help="explicit checkpoint path for resume.")
    parser.add_argument("--amp", action="store_true", help="amp training.")
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--epoch_num", default=4, type=int, help="number of epochs.")
    parser.add_argument("--subnum", default=1e4, type=float, help="sub data number.")
    parser.add_argument("--ul_data_root", required=True, type=str, help="root directory of ultrasound data.")
    parser.add_argument(
        "--base_model_path",
        required=True,
        type=str,
        help="base SonoCLIP/CLIP checkpoint path used when lora_rank=-1.",
    )
    args = parser.parse_args()

    local_rank = setup_distributed()

    trainer = CLIP_Clean_Train(
        local_rank=local_rank,
        lr=args.lr,
        weigth_decay=args.weight_decay,
        log_scale=args.log_scale,
        lora_rank=args.lora_rank,
        common_pair=args.common_pair,
        para_gamma=args.para_gamma,
        exp_name=args.exp_name,
        warmup_length=args.warmup_length,
        epoch_num=args.epoch_num,
        subnum=int(args.subnum),
        use_clip_loss=args.use_clip_loss,
        base_model_path=args.base_model_path,
    )

    trainer.train(
        common_pair=args.common_pair,
        resume=args.resume,
        amp=args.amp,
        warmup_length=args.warmup_length,
        resume_path=args.resume_path,
        data_root=args.ul_data_root,
    )
