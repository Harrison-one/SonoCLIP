import os
import sys

PROJECT_ROOT = "/dat03/sh/sh_work/CLIP/SonoCLIP"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _is_sonoclip_editable_finder(finder):
    return getattr(finder, "__module__", "").startswith("__editable___sonoclip_")


sys.meta_path[:] = [finder for finder in sys.meta_path if not _is_sonoclip_editable_finder(finder)]

import math
import cv2
import torch
import sonoclip
import argparse
import warnings
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# =========================
# Paths
# =========================
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "ViT-L-14-336px.pt")
VISUAL_CKPT_PATH = os.path.join(PROJECT_ROOT, "iter_55000_with_dwconv.pth")


# =========================
# Transforms
# 先 pad 成方形，再 resize，不做中心裁切
# =========================
PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)

clip_standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
])

res_clip_standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336), interpolation=InterpolationMode.BICUBIC),
    transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    transforms.Normalize(0.5, 0.26)
])

res_mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336), interpolation=InterpolationMode.NEAREST),
    transforms.Normalize(0.5, 0.26)
])


# =========================
# Dataset
# =========================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _find_file_by_stem(folder, stem):
    for ext in IMG_EXTS:
        p = os.path.join(folder, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def _load_split_ids(ids_file, subnum=None):
    """
    train.txt / test.txt line format:
      stem \t label_id \t class_name
    """
    items = []
    with open(ids_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"Bad line in {ids_file}: {line}")
            stem, label, class_name = parts[:3]
            items.append((stem, int(label), class_name))

    if subnum is not None:
        items = items[:int(subnum)]
    return items


class Sono_UL_Cls(Dataset):
    def __init__(
        self,
        ids_file,
        data_root,
        hi_res=True,
        subnum=None,
        use_mask=False,
        common_pair=0.0,
    ):
        self.items = _load_split_ids(ids_file, subnum=subnum)
        self.data_root = os.path.abspath(data_root)
        self.use_mask = bool(use_mask)
        # Same spirit as train_ul_1m_sig.py / Sono_UL_Txt:
        # when use_mask=True, common_pair is the probability of replacing
        # the structure mask with an all-one mask during training.
        self.common_pair = min(max(float(common_pair), 0.0), 1.0)

        if hi_res:
            self.mask_transform = res_mask_transform
            self.clip_standard_transform = res_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform

        label_to_name = {}
        for _, label, class_name in self.items:
            if label in label_to_name and label_to_name[label] != class_name:
                raise ValueError(f"Label maps to multiple class names: {label}")
            label_to_name[label] = class_name
        self.classes = [label_to_name[i] for i in sorted(label_to_name.keys())]

    def __len__(self):
        return len(self.items)

    def _read_image(self, stem, class_name):
        image_dir = os.path.join(self.data_root, class_name, "images")

        image_path = _find_file_by_stem(image_dir, stem)
        if image_path is None:
            raise FileNotFoundError(f"image not found: class={class_name}, stem={stem}")

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"cannot read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask(self, stem, class_name):
        mask_dir = os.path.join(self.data_root, class_name, "masks")

        mask_path = _find_file_by_stem(mask_dir, stem)
        if mask_path is None:
            raise FileNotFoundError(f"mask not found: class={class_name}, stem={stem}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cannot read mask: {mask_path}")
        return mask

    def _pad_to_square(self, img, mask=None):
        h, w = img.shape[:2]

        if max(h, w) == w:
            pad = (w - h) // 2
            l, r = pad, w - h - pad
            img = np.pad(
                img,
                ((l, r), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if mask is not None:
                mask = np.pad(
                    mask,
                    ((l, r), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
        else:
            pad = (h - w) // 2
            l, r = pad, h - w - pad
            img = np.pad(
                img,
                ((0, 0), (l, r), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if mask is not None:
                mask = np.pad(
                    mask,
                    ((0, 0), (l, r)),
                    mode="constant",
                    constant_values=0,
                )
        return img, mask

    def __getitem__(self, index):
        stem, label, class_name = self.items[index]
        use_real_mask = self.use_mask and (random.random() >= self.common_pair)
        img = self._read_image(stem, class_name)

        if use_real_mask:
            mask = self._read_mask(stem, class_name)

            # 尺寸不一致时旋转图像
            if mask.shape != img.shape[:2]:
                img = np.rot90(img)
                if mask.shape != img.shape[:2]:
                    raise RuntimeError(
                        f"image/mask shape mismatch after rotation: "
                        f"stem={stem}, img={img.shape[:2]}, mask={mask.shape}"
                    )

            img, mask = self._pad_to_square(img, mask)
        else:
            img, _ = self._pad_to_square(img)
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

        image_torch = self.clip_standard_transform(img)

        if use_real_mask:
            mask_torch = self.mask_transform(mask)
        else:
            mask_torch = self.mask_transform(mask)

        return image_torch, mask_torch, label


# =========================
# Scheduler
# =========================
class WarmupCosineScheduler:
    def __init__(self, optimizer, base_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.total_steps = max(int(total_steps), 1)

    def step(self, step_idx):
        step_idx = int(step_idx)

        if self.warmup_steps > 0 and step_idx < self.warmup_steps:
            lr = self.base_lr * float(step_idx + 1) / float(self.warmup_steps)
        else:
            if self.total_steps <= self.warmup_steps:
                progress = 1.0
            else:
                progress = float(step_idx - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
                progress = min(max(progress, 0.0), 1.0)
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        for group in self.optimizer.param_groups:
            group["lr"] = lr


# =========================
# Trainer
# 单卡 / 单进程
# 每个 epoch 验证一次
# =========================
class SonoCLIPLinearProbeSingle:
    def __init__(
        self,
        device="cuda",
        lr=1e-3,
        weight_decay=1e-2,
        exp_name="auto",
        warmup_length=200,
        epoch_num=10,
        batch_size=8,
        use_mask=False,
        num_workers_train=8,
        num_workers_test=8,
        data_root="",
        train_txt="",
        test_txt="",
        subnum=None,
        hi_res=True,
        common_pair=0.1,
        base_model_path=BASE_MODEL_PATH,
        alpha_vision_ckpt=VISUAL_CKPT_PATH,
        save_dir="",
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_length = int(warmup_length)
        self.num_epoch = int(epoch_num)
        self.batch_size = int(batch_size)
        self.use_mask = bool(use_mask)
        self.num_workers_train = int(num_workers_train)
        self.num_workers_test = int(num_workers_test)
        self.data_root = data_root
        self.train_txt = train_txt
        self.test_txt = test_txt
        self.subnum = subnum
        self.hi_res = bool(hi_res)
        self.common_pair = min(max(float(common_pair), 0.0), 1.0)
        self.base_model_path = base_model_path
        self.alpha_vision_ckpt = alpha_vision_ckpt
        self.save_dir = save_dir
        self.class_names = None

        checkpoint_path = self.alpha_vision_ckpt

        # 基础模型
        if not os.path.isfile(self.base_model_path):
            raise FileNotFoundError(f"base model not found: {self.base_model_path}")
        if not self.hi_res and os.path.basename(self.base_model_path) == "ViT-L-14-336px.pt":
            raise ValueError("--no_hi_res requires a local 224px ViT-L/14 checkpoint, not ViT-L-14-336px.pt")
        self.model, _ = sonoclip.load(self.base_model_path, device="cpu", lora_adapt=False, rank=-1)

        # 加载 visual 权重
        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if "visual" in ckpt:
                    visual_state_dict = ckpt["visual"]
                    if any(k.startswith("module.") for k in visual_state_dict.keys()):
                        visual_state_dict = {k.replace("module.", ""): v for k, v in visual_state_dict.items()}
                    load_msg = self.model.visual.load_state_dict(visual_state_dict, strict=False)
                    print(f"Successfully loaded visual encoder from checkpoint: {checkpoint_path}")
                    print(f"Checkpoint step: {ckpt.get('step', 'unknown')}")
                    print(f"Missing keys: {len(load_msg.missing_keys)}, Unexpected keys: {len(load_msg.unexpected_keys)}")
                else:
                    load_msg = self.model.visual.load_state_dict(ckpt, strict=False)
                    print(f"Successfully loaded visual encoder weights from: {checkpoint_path}")
                    print(f"Missing keys: {len(load_msg.missing_keys)}, Unexpected keys: {len(load_msg.unexpected_keys)}")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
                print("Using base model weights instead.")
        else:
            print(f"Warning: Checkpoint file not found: {checkpoint_path}")
            print("Using base model weights instead.")

        self.model = self.model.float().to(self.device)

        # 冻结 backbone
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.cls_head = None

        if self.save_dir:
            self.logdir = os.path.abspath(self.save_dir)
        elif exp_name == "auto":
            self.logdir = (
                f"log/sonoclip_linear_probe_single/"
                f"lr={self.lr}_wd={self.weight_decay}_"
                f"use_mask={int(self.use_mask)}_cp={self.common_pair}_"
                f"epochs={self.num_epoch}_bs={self.batch_size}"
            )
        else:
            self.logdir = exp_name

        self.ckptdir = os.path.join(self.logdir, "ckpt")
        os.makedirs(self.ckptdir, exist_ok=True)
        self.test_results_txt = os.path.join(self.logdir, "test_results.txt")
        self.writer = SummaryWriter(self.logdir)
        print(f"Save directory: {self.logdir}")

        self.optimizer = None
        self.scheduler = None
        self.num_classes = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

    def _build_head_and_optimizer(self, num_classes, total_steps):
        self.num_classes = int(num_classes)
        in_dim = self.model.text_projection.shape[1]

        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_dim, self.num_classes),
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.cls_head.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            base_lr=self.lr,
            warmup_steps=self.warmup_length,
            total_steps=total_steps,
        )

    def encode_image(self, images, masks):
        with torch.no_grad():
            feat = self.model.visual(images, masks)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return feat

    def inference(self, images, masks, labels):
        feat = self.encode_image(images, masks)
        logits = self.cls_head(feat)
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    def _append_test_result(self, epoch, test_name, v1, v2):
        write_header = not os.path.isfile(self.test_results_txt) or os.path.getsize(self.test_results_txt) == 0
        with open(self.test_results_txt, "a", encoding="utf-8") as f:
            if write_header:
                f.write("epoch\ttest_name\tvalue1\tvalue2\n")
            f.write(f"{epoch}\t{test_name}\t{v1:.6f}\t{v2:.6f}\n")
            f.flush()

    def _append_per_class_result(self, epoch, per_class_stats, class_names, eval_name="test"):
        per_class_txt = os.path.join(self.logdir, f"per_class_results_{eval_name}.txt")
        write_header = not os.path.isfile(per_class_txt) or os.path.getsize(per_class_txt) == 0

        with open(per_class_txt, "a", encoding="utf-8") as f:
            if write_header:
                f.write("epoch\tclass_id\tclass_name\tsamples\tacc\tprecision\trecall\tf1\n")

            for cid in range(len(class_names)):
                stat = per_class_stats[cid]
                f.write(
                    f"{epoch}\t{cid}\t{class_names[cid]}\t"
                    f"{stat['support']}\t{stat['acc']:.6f}\t{stat['precision']:.6f}\t"
                    f"{stat['recall']:.6f}\t{stat['f1']:.6f}\n"
                )
            f.flush()

    def _compute_metrics_from_confmat(self, confmat):
        """
        confmat[i, j]: true=i, pred=j
        """
        num_classes = confmat.shape[0]
        total = confmat.sum()
        correct = np.trace(confmat)

        accuracy = float(correct) / float(max(total, 1))

        per_class_stats = {}
        precisions, recalls, f1s = [], [], []

        for c in range(num_classes):
            tp = confmat[c, c]
            fn = confmat[c, :].sum() - tp
            fp = confmat[:, c].sum() - tp
            support = confmat[c, :].sum()

            class_acc = float(tp) / float(max(support, 1))
            precision = float(tp) / float(max(tp + fp, 1))
            recall = float(tp) / float(max(tp + fn, 1))
            if precision + recall > 0:
                f1 = 2.0 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            per_class_stats[c] = {
                "support": int(support),
                "acc": class_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            if support > 0:
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

        macro_precision = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
        macro_recall = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
        macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0

        micro_precision = accuracy
        micro_recall = accuracy
        micro_f1 = accuracy

        return {
            "accuracy": accuracy,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "per_class_stats": per_class_stats,
        }

    def _save_confusion_matrix(self, confmat, class_names, epoch, normalize=False, eval_name="test"):
        save_name = f"confusion_matrix_{eval_name}_epoch_{epoch}"
        if normalize:
            save_name += "_norm"

        png_path = os.path.join(self.logdir, save_name + ".png")
        npy_path = os.path.join(self.logdir, save_name + ".npy")
        np.save(npy_path, confmat)

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Warning: failed to import matplotlib, skip confusion matrix png: {e}")
            return

        mat = confmat.astype(np.float64).copy()
        if normalize:
            row_sum = mat.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            mat = mat / row_sum

        plt.figure(figsize=(max(8, len(class_names) * 1.2), max(6, len(class_names) * 1.0)))
        plt.imshow(mat, interpolation="nearest", cmap="Blues")
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)

        plt.title("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        thresh = mat.max() / 2.0 if mat.size > 0 else 0.5
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                text_val = f"{mat[i, j]:.2f}" if normalize else f"{int(mat[i, j])}"
                plt.text(
                    j,
                    i,
                    text_val,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if mat[i, j] > thresh else "black",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()

    @torch.no_grad()
    def test_epoch(self, dataloader):
        self.model.eval()
        self.cls_head.eval()

        confmat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        total = 0
        top1 = 0
        top5 = 0

        for images, masks, labels in tqdm(dataloader, desc="Testing", leave=False):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()

            feat = self.encode_image(images, masks)
            logits = self.cls_head(feat)

            pred1 = logits.topk(1, dim=1)[1].squeeze(1)
            k5 = min(5, logits.shape[1])
            pred5 = logits.topk(k5, dim=1)[1]

            bs = labels.size(0)
            total += bs
            top1 += (pred1 == labels).sum().item()

            for i in range(bs):
                y_true = labels[i].item()
                y_pred = pred1[i].item()
                confmat[y_true, y_pred] += 1

                if y_true in pred5[i].tolist():
                    top5 += 1

        return {
            "total": total,
            "top1": top1,
            "top5": top5,
            "confmat": confmat,
        }

    def evaluate(self, test_loader, epoch, class_names=None, save_confmat=False, eval_name="test"):
        result = self.test_epoch(test_loader)

        total = result["total"]
        top1 = result["top1"]
        top5 = result["top5"]
        confmat = result["confmat"]

        metrics = self._compute_metrics_from_confmat(confmat)

        accuracy = metrics["accuracy"]
        micro_f1 = metrics["micro_f1"]
        macro_f1 = metrics["macro_f1"]
        macro_precision = metrics["macro_precision"]
        macro_recall = metrics["macro_recall"]
        per_class_stats = metrics["per_class_stats"]

        top5_acc = float(top5) / float(max(total, 1))

        if class_names is None:
            class_names = [str(i) for i in range(len(per_class_stats))]

        print("=====================================")
        print(f"{eval_name} epoch {epoch}")
        print(f"accuracy(top1): {accuracy:.6f}")
        print(f"top1_acc_raw: {float(top1) / float(max(total, 1)):.6f}")
        print(f"top5_acc: {top5_acc:.6f}")
        print(f"micro_f1: {micro_f1:.6f}")
        print(f"macro_f1: {macro_f1:.6f}")
        print(f"macro_precision: {macro_precision:.6f}")
        print(f"macro_recall: {macro_recall:.6f}")
        print("per-class results:")
        for cid in range(len(per_class_stats)):
            stat = per_class_stats[cid]
            cname = class_names[cid]
            print(
                f"  [{cid}] {cname}: "
                f"samples={stat['support']}, "
                f"acc={stat['acc']:.6f}, "
                f"precision={stat['precision']:.6f}, "
                f"recall={stat['recall']:.6f}, "
                f"f1={stat['f1']:.6f}"
            )
        print("=====================================")

        self.writer.add_scalar(f"{eval_name}/accuracy", accuracy, epoch)
        self.writer.add_scalar(f"{eval_name}/top5_acc", top5_acc, epoch)
        self.writer.add_scalar(f"{eval_name}/micro_f1", micro_f1, epoch)
        self.writer.add_scalar(f"{eval_name}/macro_f1", macro_f1, epoch)
        self.writer.add_scalar(f"{eval_name}/macro_precision", macro_precision, epoch)
        self.writer.add_scalar(f"{eval_name}/macro_recall", macro_recall, epoch)

        for cid in range(len(per_class_stats)):
            stat = per_class_stats[cid]
            self.writer.add_scalar(f"{eval_name}_per_class/class_{cid}_acc", stat["acc"], epoch)
            self.writer.add_scalar(f"{eval_name}_per_class/class_{cid}_f1", stat["f1"], epoch)
            self.writer.add_scalar(f"{eval_name}_per_class/class_{cid}_precision", stat["precision"], epoch)
            self.writer.add_scalar(f"{eval_name}_per_class/class_{cid}_recall", stat["recall"], epoch)

        self._append_test_result(epoch, f"{eval_name}_accuracy_top1_top5", accuracy, top5_acc)
        self._append_test_result(epoch, f"{eval_name}_f1_micro_macro", micro_f1, macro_f1)
        self._append_test_result(epoch, f"{eval_name}_precision_recall_macro", macro_precision, macro_recall)
        self._append_per_class_result(epoch, per_class_stats, class_names, eval_name=eval_name)

        if save_confmat:
            self._save_confusion_matrix(confmat, class_names, epoch, normalize=False, eval_name=eval_name)
            self._save_confusion_matrix(confmat, class_names, epoch, normalize=True, eval_name=eval_name)

        self.model.eval()
        self.cls_head.train()

    def save_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "num_classes": self.num_classes,
                "use_mask": self.use_mask,
                "common_pair": self.common_pair,
                "alpha_vision_ckpt": self.alpha_vision_ckpt,
                "class_names": self.class_names,
                "cls_head": self.cls_head.state_dict(),
                "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            },
            os.path.join(self.ckptdir, f"epoch_{epoch}.pth"),
        )

    def load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.cls_head.load_state_dict(ckpt["cls_head"], strict=True)
        if self.optimizer is not None and ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.class_names = ckpt.get("class_names", self.class_names)
        return int(ckpt.get("epoch", 0))

    def train(self, resume=False, amp=False):
        trainset = Sono_UL_Cls(
            ids_file=self.train_txt,
            data_root=self.data_root,
            hi_res=self.hi_res,
            subnum=self.subnum,
            use_mask=self.use_mask,
            common_pair=self.common_pair if self.use_mask else 1.0,
        )
        testset_real_mask = Sono_UL_Cls(
            ids_file=self.test_txt,
            data_root=self.data_root,
            hi_res=self.hi_res,
            subnum=None,
            use_mask=self.use_mask,
            common_pair=0.0,
        )
        testset_all_one = Sono_UL_Cls(
            ids_file=self.test_txt,
            data_root=self.data_root,
            hi_res=self.hi_res,
            subnum=None,
            use_mask=False,
            common_pair=1.0,
        )

        if trainset.classes != testset_real_mask.classes:
            raise RuntimeError(
                f"Train/Test classes mismatch:\ntrain={trainset.classes}\ntest={testset_real_mask.classes}"
            )

        self.class_names = trainset.classes

        train_loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers_train,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )
        test_loader_real_mask = DataLoader(
            testset_real_mask,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers_test,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )
        test_loader_all_one = DataLoader(
            testset_all_one,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers_test,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )
        test_loaders = {"test_all_one": test_loader_all_one}
        if self.use_mask:
            test_loaders["test_real_mask"] = test_loader_real_mask

        total_steps = len(train_loader) * self.num_epoch
        self._build_head_and_optimizer(num_classes=len(trainset.classes), total_steps=total_steps)

        start_epoch = 0
        global_step = 0

        if resume and os.path.isdir(self.ckptdir):
            ckpt_files = [f for f in os.listdir(self.ckptdir) if f.startswith("epoch_") and f.endswith(".pth")]
            if ckpt_files:
                ckpt_files = sorted(ckpt_files, key=lambda x: int(x[6:-4]))
                resume_pth = os.path.join(self.ckptdir, ckpt_files[-1])
                loaded_epoch = self.load_checkpoint(resume_pth)
                start_epoch = loaded_epoch
                global_step = loaded_epoch * max(len(train_loader), 1)
                print(f"load resumed checkpoint: {resume_pth}")

        for epoch in range(start_epoch, self.num_epoch):
            self.model.eval()
            self.cls_head.train()

            epoch_loss_sum = 0.0
            epoch_acc_sum = 0.0
            epoch_batch_count = 0

            for i, (images, masks, labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch + 1}")):
                step = global_step + i
                self.scheduler.step(step)

                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()

                self.optimizer.zero_grad(set_to_none=True)

                use_amp = amp and self.device.type == "cuda"
                if use_amp:
                    with torch.cuda.amp.autocast():
                        loss, logits = self.inference(images, masks, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss, logits = self.inference(images, masks, labels)
                    loss.backward()
                    self.optimizer.step()

                with torch.no_grad():
                    pred1 = logits.topk(1, dim=1)[1].squeeze(1)
                    batch_acc = (pred1 == labels).float().mean().item()

                epoch_loss_sum += loss.item()
                epoch_acc_sum += batch_acc
                epoch_batch_count += 1

            global_step += len(train_loader)

            avg_train_loss = epoch_loss_sum / max(epoch_batch_count, 1)
            avg_train_acc = epoch_acc_sum / max(epoch_batch_count, 1)

            self.writer.add_scalar("train/epoch_loss", avg_train_loss, epoch + 1)
            self.writer.add_scalar("train/epoch_acc1", avg_train_acc, epoch + 1)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch + 1)

            print("=====================================")
            print(f"epoch {epoch + 1}/{self.num_epoch} finished")
            print(f"use_mask: {self.use_mask}")
            print(f"common_pair(all-one prob in train): {self.common_pair if self.use_mask else 1.0}")
            print(f"lr: {self.optimizer.param_groups[0]['lr']:.8f}")
            print(f"train loss: {avg_train_loss:.6f}")
            print(f"train acc-1: {avg_train_acc:.6f}")
            print("=====================================")

            # 每个 epoch 验证一次
            for eval_name, test_loader in test_loaders.items():
                self.evaluate(
                    test_loader,
                    epoch + 1,
                    class_names=self.class_names,
                    save_confmat=False,
                    eval_name=eval_name,
                )

            # 每个 epoch 保存一次
            self.save_checkpoint(epoch + 1)

        # 最后一轮额外保存混淆矩阵
        for eval_name, test_loader in test_loaders.items():
            self.evaluate(
                test_loader,
                self.num_epoch,
                class_names=self.class_names,
                save_confmat=True,
                eval_name=eval_name,
            )


# =========================
# Main
# =========================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="SonoCLIP linear probe with train-time mixed real/all-one masks."
    )
    parser.add_argument(
        "--data_root",
        default="/dat03/sh/sh_work/CLIP/zero_shot_six_plane/test_cls_plane",
        type=str,
        help="root containing class folders",
    )
    parser.add_argument(
        "--train_txt",
        default="/dat03/sh/sh_work/CLIP/zero_shot_six_plane/test_cls_plane_split/train.txt",
        type=str,
        help="train split txt",
    )
    parser.add_argument(
        "--test_txt",
        default="/dat03/sh/sh_work/CLIP/zero_shot_six_plane/test_cls_plane_split/test.txt",
        type=str,
        help="test split txt",
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="classifier head learning rate")
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="weight decay")
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    parser.add_argument("--amp", action="store_true", help="use mixed precision")
    parser.add_argument("--exp_name", default="auto", type=str, help="experiment name")
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
        help="explicit output directory for ckpt, TensorBoard logs, test results, and confusion matrices",
    )
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup steps")
    parser.add_argument("--epoch_num", default=50, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--use_mask", action="store_true", help="use real structure masks; otherwise all-one masks")
    parser.add_argument("--base_model_path", default=BASE_MODEL_PATH, type=str, help="local base ViT/CLIP checkpoint path")
    parser.add_argument(
        "--common_pair",
        default=0.1,
        type=float,
        help="train-time probability of replacing a real mask with an all-one mask when --use_mask is set",
    )
    parser.add_argument(
        "--alpha_vision_ckpt",
        default=VISUAL_CKPT_PATH,
        type=str,
        help="SonoCLIP visual checkpoint path",
    )
    parser.add_argument("--subnum", default=None, type=int, help="optional subset of train split")
    parser.add_argument("--no_hi_res", action="store_true", help="use 224 instead of 336")
    parser.add_argument("--num_workers_train", default=8, type=int, help="train dataloader workers")
    parser.add_argument("--num_workers_test", default=8, type=int, help="test dataloader workers")
    parser.add_argument("--device", default="cuda", type=str, help="cuda or cpu")
    args = parser.parse_args()

    trainer = SonoCLIPLinearProbeSingle(
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        exp_name=args.exp_name,
        save_dir=args.save_dir,
        warmup_length=args.warmup_length,
        epoch_num=args.epoch_num,
        batch_size=args.batch_size,
        use_mask=args.use_mask,
        num_workers_train=args.num_workers_train,
        num_workers_test=args.num_workers_test,
        data_root=args.data_root,
        train_txt=args.train_txt,
        test_txt=args.test_txt,
        subnum=args.subnum,
        hi_res=(not args.no_hi_res),
        common_pair=args.common_pair,
        base_model_path=args.base_model_path,
        alpha_vision_ckpt=args.alpha_vision_ckpt,
    )

    trainer.train(
        resume=args.resume,
        amp=args.amp,
    )
