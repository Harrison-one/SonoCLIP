import os
import math
import cv2
import torch
import sonoclip
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAIN_ROOT)

DEFAULT_SEG_DATA_ROOT = os.path.join(PROJECT_ROOT, "seg_data", "test_seg_plane")
DEFAULT_SEG_SPLIT_ROOT = os.path.join(PROJECT_ROOT, "seg_data", "test_seg_plane_split")
DEFAULT_BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "ViT-L-14-336px.pt")
DEFAULT_VISION_CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "sonoclip_vision.pth")


# =========================
# Transforms
# =========================
PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)

# SonoCLIP README 里 alpha 的常用归一化口径
ALPHA_MEAN = 0.5
ALPHA_STD = 0.26

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def build_img_transform(img_size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
    ])


def build_mask_transform(img_size):
    return transforms.Compose([
        transforms.ToTensor(),  # -> [1,H,W], 0~1
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
    ])


# =========================
# Utils
# =========================
def _find_file_by_stem(folder, stem):
    for ext in IMG_EXTS:
        p = os.path.join(folder, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def _normalize_optional_path(path):
    if path is None:
        return None
    path = str(path)
    if path.strip().lower() in ("", "none", "null"):
        return None
    return path


def _load_split_ids(ids_file, subnum=None):
    """
    train.txt / test.txt:
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


def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def batch_dice_iou(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.flatten(1)
    targets = targets.flatten(1)

    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter

    dice = (2.0 * inter + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)
    iou = (inter + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()


@torch.no_grad()
def samplewise_dice_iou(logits, targets, threshold=0.5, eps=1e-6):
    """
    return:
        dice: [B]
        iou:  [B]
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.flatten(1)
    targets = targets.flatten(1)

    inter = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum - inter

    dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
    iou = (inter + eps) / (union + eps)

    return dice, iou


# =========================
# Dataset
# 每张图只做一个前景目标的二值分割
# 但整个数据集可以有多个类别
# alpha 不使用 GT mask，使用“符合 SonoCLIP 预期分布”的全 1 alpha
# =========================
class SonoULSegDataset(Dataset):
    def __init__(
        self,
        ids_file,
        data_root,
        img_size=336,
        subnum=None,
    ):
        self.items = _load_split_ids(ids_file, subnum=subnum)
        self.data_root = os.path.abspath(data_root)
        self.img_transform = build_img_transform(img_size)
        self.mask_transform = build_mask_transform(img_size)
        self.img_size = int(img_size)

        label_to_name = {}
        for _, label, class_name in self.items:
            if label in label_to_name and label_to_name[label] != class_name:
                raise ValueError(f"Label maps to multiple class names: {label}")
            label_to_name[label] = class_name
        self.classes = [label_to_name[i] for i in sorted(label_to_name.keys())]

    def __len__(self):
        return len(self.items)

    def _read_image_and_mask(self, stem, class_name):
        image_dir = os.path.join(self.data_root, class_name, "images")
        mask_dir = os.path.join(self.data_root, class_name, "masks")

        image_path = _find_file_by_stem(image_dir, stem)
        if image_path is None:
            raise FileNotFoundError(f"image not found: class={class_name}, stem={stem}")

        mask_path = _find_file_by_stem(mask_dir, stem)
        if mask_path is None:
            raise FileNotFoundError(f"mask not found: class={class_name}, stem={stem}")

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"cannot read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cannot read mask: {mask_path}")

        return img, mask

    def __getitem__(self, index):
        stem, label, class_name = self.items[index]
        img, mask = self._read_image_and_mask(stem, class_name)

        # 若尺寸不匹配，尝试旋转图像
        if mask.shape != img.shape[:2]:
            img = np.rot90(img)
            if mask.shape != img.shape[:2]:
                raise RuntimeError(
                    f"image/mask shape mismatch after rotation: "
                    f"stem={stem}, img={img.shape[:2]}, mask={mask.shape}"
                )

        # pad to square
        h, w = img.shape[:2]
        if h < w:
            pad = (w - h) // 2
            l, r = pad, w - h - pad
            img = np.pad(img, ((l, r), (0, 0), (0, 0)), mode="constant", constant_values=0)
            mask = np.pad(mask, ((l, r), (0, 0)), mode="constant", constant_values=0)
        elif w < h:
            pad = (h - w) // 2
            l, r = pad, h - w - pad
            img = np.pad(img, ((0, 0), (l, r), (0, 0)), mode="constant", constant_values=0)
            mask = np.pad(mask, ((0, 0), (l, r)), mode="constant", constant_values=0)

        image_torch = self.img_transform(img)          # [3,S,S]
        mask_torch = self.mask_transform(mask).float() # [1,S,S], 0~1
        mask_torch = (mask_torch > 0.5).float()        # 二值化

        # 分割任务里不能把 GT mask 喂回 backbone，避免标签泄漏
        # 这里使用“全 1 alpha”，并按 SonoCLIP 预期分布归一化：
        # alpha=1 -> (1 - 0.5) / 0.26
        alpha_ones = torch.ones_like(mask_torch)
        alpha_ones = (alpha_ones - ALPHA_MEAN) / ALPHA_STD

        return image_torch, alpha_ones, mask_torch, label, stem, class_name


# =========================
# LR Scheduler
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
# Frozen SonoCLIP ViT dense feature extractor
# 按 SonoCLIP 实际视觉前向，把 alpha 分支接进去
# 输出最后一层 patch token 对应的 dense feature map
# =========================
class FrozenSonoCLIPDenseEncoder(nn.Module):
    def __init__(self, model, img_size=336):
        super().__init__()
        self.model = model
        self.visual = model.visual
        self.img_size = int(img_size)

        v = self.visual

        if not hasattr(v, "conv1") or not hasattr(v, "conv1_alpha"):
            raise RuntimeError(
                "Current SonoCLIP visual does not expose conv1 / conv1_alpha. "
                "This encoder assumes SonoCLIP ViT backbone."
            )

        self.patch_size = v.conv1.kernel_size[0]
        self.width = v.conv1.out_channels

        if not hasattr(v, "class_embedding"):
            raise RuntimeError("visual.class_embedding not found.")
        if not hasattr(v, "positional_embedding"):
            raise RuntimeError("visual.positional_embedding not found.")
        if not hasattr(v, "ln_pre"):
            raise RuntimeError("visual.ln_pre not found.")
        if not hasattr(v, "transformer"):
            raise RuntimeError("visual.transformer not found.")

    @torch.no_grad()
    def forward(self, images, alpha):
        """
        images: [B,3,H,W]
        alpha : [B,1,H,W]

        return:
            feat_map: [B, C, Gh, Gw]
        """
        v = self.visual

        if alpha is None:
            raise ValueError("SonoCLIP visual forward assumes alpha is not None.")

        images = images.to(dtype=v.conv1.weight.dtype)
        alpha = alpha.to(dtype=v.conv1_alpha.weight.dtype)

        if alpha.dim() != 4:
            raise ValueError(f"alpha must be 4D [B,1,H,W], got shape={tuple(alpha.shape)}")
        if alpha.shape[1] != 1:
            raise ValueError(f"alpha channel must be 1, got shape={tuple(alpha.shape)}")

        if alpha.shape[-2:] != images.shape[-2:]:
            alpha = F.interpolate(
                alpha,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # SonoCLIP 实际 patch embedding
        x = v.conv1(images) + v.conv1_alpha(alpha)   # [B, C, Gh, Gw]
        B, C, Gh, Gw = x.shape

        # [B, C, Gh, Gw] -> [B, N, C]
        x = x.reshape(B, C, Gh * Gw).permute(0, 2, 1)

        # cls token
        cls = v.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, C]

        # positional embedding
        pos = v.positional_embedding.to(x.dtype)
        if pos.shape[0] != x.shape[1]:
            raise RuntimeError(
                f"Positional embedding length mismatch: "
                f"pos={pos.shape[0]}, tokens={x.shape[1]}"
            )
        x = x + pos

        # ln_pre
        x = v.ln_pre(x)

        # transformer
        x = x.permute(1, 0, 2)   # [L, B, C]
        x = v.transformer(x)
        x = x.permute(1, 0, 2)   # [B, 1+N, C]

        # 去掉 cls token，取最后一层 patch tokens
        x = x[:, 1:, :]          # [B, N, C]

        # reshape 回空间图
        feat_map = x.permute(0, 2, 1).reshape(B, C, Gh, Gw)   # [B, C, Gh, Gw]
        return feat_map


# =========================
# Lightweight decoder
# =========================
class SimpleSegDecoder(nn.Module):
    def __init__(self, in_channels, mid_channels=512):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels // 2, mid_channels // 4, 3, padding=1),
            nn.BatchNorm2d(mid_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels // 4, 1, 1),
        )

    def forward(self, feat_map, out_size):
        x = self.block(feat_map)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


# =========================
# Full model
# =========================
class SonoCLIPFrozenSegModel(nn.Module):
    def __init__(self, sonoclip_model, img_size=336, decoder_mid_channels=512):
        super().__init__()
        self.encoder = FrozenSonoCLIPDenseEncoder(sonoclip_model, img_size=img_size)
        self.decoder = SimpleSegDecoder(
            in_channels=self.encoder.width,
            mid_channels=decoder_mid_channels,
        )

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def forward(self, images, alpha):
        with torch.no_grad():
            feat_map = self.encoder(images, alpha)
        logits = self.decoder(feat_map, out_size=images.shape[-2:])
        return logits


# =========================
# Trainer
# 每个 epoch 验证一次、保存一次
# 并输出每个类别的 Dice / IoU
# 最后一个 epoch 保存分割结果
# =========================
class SonoCLIPSegTrainer:
    def __init__(
        self,
        device="cuda",
        lr=1e-3,
        weight_decay=1e-2,
        exp_name="auto",
        warmup_length=200,
        epoch_num=10,
        batch_size=4,
        num_workers_train=8,
        num_workers_test=8,
        data_root="",
        train_txt="",
        test_txt="",
        subnum=None,
        hi_res=True,
        bce_weight=0.5,
        save_threshold=0.5,
        decoder_mid_channels=512,
        base_model_path=DEFAULT_BASE_MODEL_PATH,
        vision_ckpt=DEFAULT_VISION_CKPT_PATH,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_length = int(warmup_length)
        self.num_epoch = int(epoch_num)
        self.batch_size = int(batch_size)
        self.num_workers_train = int(num_workers_train)
        self.num_workers_test = int(num_workers_test)
        self.data_root = data_root
        self.train_txt = train_txt
        self.test_txt = test_txt
        self.subnum = subnum
        self.hi_res = bool(hi_res)
        self.img_size = 336 if self.hi_res else 224
        self.bce_weight = float(bce_weight)
        self.save_threshold = float(save_threshold)
        self.decoder_mid_channels = int(decoder_mid_channels)
        self.base_model_path = base_model_path
        self.vision_ckpt = _normalize_optional_path(vision_ckpt)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.class_names = None

        base_model, _ = sonoclip.load(
            self.base_model_path if self.hi_res else "ViT-L/14",
            device="cpu",
            lora_adapt=False,
            rank=-1,
        )

        if self.vision_ckpt and os.path.exists(self.vision_ckpt):
            try:
                ckpt = torch.load(self.vision_ckpt, map_location="cpu", weights_only=False)
                if "visual" in ckpt:
                    visual_state_dict = ckpt["visual"]
                    if any(k.startswith("module.") for k in visual_state_dict.keys()):
                        visual_state_dict = {k.replace("module.", ""): v for k, v in visual_state_dict.items()}
                    load_msg = base_model.visual.load_state_dict(visual_state_dict, strict=False)
                    print(f"Loaded visual encoder from checkpoint: {self.vision_ckpt}")
                    print(f"Missing keys: {len(load_msg.missing_keys)}, Unexpected keys: {len(load_msg.unexpected_keys)}")
                else:
                    load_msg = base_model.visual.load_state_dict(ckpt, strict=False)
                    print(f"Loaded visual weights from checkpoint: {self.vision_ckpt}")
                    print(f"Missing keys: {len(load_msg.missing_keys)}, Unexpected keys: {len(load_msg.unexpected_keys)}")
            except Exception as e:
                print(f"Warning: failed to load checkpoint {self.vision_ckpt}: {e}")
                print("Using base SonoCLIP weights.")
        else:
            print(f"Warning: checkpoint not found: {self.vision_ckpt}")
            print("Using base SonoCLIP weights.")

        self.base_model = base_model.float()

        if exp_name == "auto":
            self.logdir = (
                f"log/sonoclip_frozen_seg_realalpha/"
                f"lr={self.lr}_wd={self.weight_decay}_epochs={self.num_epoch}_"
                f"bs={self.batch_size}_mid={self.decoder_mid_channels}_size={self.img_size}"
            )
        else:
            self.logdir = exp_name

        self.ckptdir = os.path.join(self.logdir, "ckpt")
        os.makedirs(self.ckptdir, exist_ok=True)
        self.test_results_path = os.path.join(self.logdir, "test_results.txt")
        self.writer = SummaryWriter(self.logdir)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

    def build_model_and_optimizer(self):
        self.model = SonoCLIPFrozenSegModel(
            self.base_model,
            img_size=self.img_size,
            decoder_mid_channels=self.decoder_mid_channels,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.decoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def compute_loss(self, logits, masks):
        loss_bce = F.binary_cross_entropy_with_logits(logits, masks)
        loss_dice = dice_loss_with_logits(logits, masks)
        loss = self.bce_weight * loss_bce + (1.0 - self.bce_weight) * loss_dice
        return loss, loss_bce, loss_dice

    def _save_segmentation_results(self, logits, masks, stems, class_names_batch, epoch_idx):
        """
        保存预测结果和 GT：
          logdir/seg_results_epoch_{epoch}/class_name/stem_pred.png
          logdir/seg_results_epoch_{epoch}/class_name/stem_gt.png
        """
        save_root = os.path.join(self.logdir, f"seg_results_epoch_{epoch_idx}")
        os.makedirs(save_root, exist_ok=True)

        probs = torch.sigmoid(logits)
        preds = (probs > self.save_threshold).float()

        preds = preds.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()

        for i in range(preds.shape[0]):
            cls_name = class_names_batch[i]
            stem = stems[i]

            cls_dir = os.path.join(save_root, cls_name)
            os.makedirs(cls_dir, exist_ok=True)

            pred_mask = (preds[i, 0] * 255).astype(np.uint8)
            gt_mask = (masks[i, 0] * 255).astype(np.uint8)

            pred_path = os.path.join(cls_dir, f"{stem}_pred.png")
            gt_path = os.path.join(cls_dir, f"{stem}_gt.png")

            cv2.imwrite(pred_path, pred_mask)
            cv2.imwrite(gt_path, gt_mask)

    def _append_test_results(self, results):
        os.makedirs(os.path.dirname(self.test_results_path) or ".", exist_ok=True)
        with open(self.test_results_path, "a", encoding="utf-8") as f:
            f.write("=====================================\n")
            f.write(f"epoch {results['epoch']}\n")
            f.write(f"val loss: {results['loss']:.6f}\n")
            f.write(f"val bce: {results['bce']:.6f}\n")
            f.write(f"val dice_loss: {results['dice_loss']:.6f}\n")
            f.write(f"val dice: {results['dice']:.6f}\n")
            f.write(f"val iou: {results['iou']:.6f}\n")
            f.write("per-class val metrics:\n")
            for item in results["per_class"]:
                f.write(
                    f"  [{item['class_id']}] {item['class_name']}: "
                    f"samples={item['samples']}, "
                    f"dice={item['dice']:.6f}, "
                    f"iou={item['iou']:.6f}\n"
                )
            f.write("=====================================\n\n")

    @torch.no_grad()
    def evaluate(self, test_loader, epoch_idx, save_results=False):
        self.model.eval()

        total_loss = 0.0
        total_bce = 0.0
        total_dice_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_n = 0

        # per-class 统计
        per_class_sum = {}
        per_class_count = {}

        for images, alpha_ones, masks, labels, stems, class_names_batch in tqdm(
            test_loader, desc=f"Val Epoch {epoch_idx}", leave=False
        ):
            images = images.to(self.device, non_blocking=True).float()
            alpha_ones = alpha_ones.to(self.device, non_blocking=True).float()
            masks = masks.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()

            logits = self.model(images, alpha_ones)
            loss, loss_bce, loss_dice = self.compute_loss(logits, masks)

            dice, iou = batch_dice_iou(logits, masks)
            dice_each, iou_each = samplewise_dice_iou(logits, masks)

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_bce += loss_bce.item() * bs
            total_dice_loss += loss_dice.item() * bs
            total_dice += dice * bs
            total_iou += iou * bs
            total_n += bs

            # per-class 累加
            for i in range(bs):
                cls_id = int(labels[i].item())
                if cls_id not in per_class_sum:
                    per_class_sum[cls_id] = {
                        "dice": 0.0,
                        "iou": 0.0,
                    }
                    per_class_count[cls_id] = 0

                per_class_sum[cls_id]["dice"] += float(dice_each[i].item())
                per_class_sum[cls_id]["iou"] += float(iou_each[i].item())
                per_class_count[cls_id] += 1

            if save_results:
                self._save_segmentation_results(
                    logits=logits,
                    masks=masks,
                    stems=stems,
                    class_names_batch=class_names_batch,
                    epoch_idx=epoch_idx,
                )

        avg_loss = total_loss / max(total_n, 1)
        avg_bce = total_bce / max(total_n, 1)
        avg_dice_loss = total_dice_loss / max(total_n, 1)
        avg_dice = total_dice / max(total_n, 1)
        avg_iou = total_iou / max(total_n, 1)

        print("=====================================")
        print(f"epoch {epoch_idx}")
        print(f"val loss: {avg_loss:.6f}")
        print(f"val bce: {avg_bce:.6f}")
        print(f"val dice_loss: {avg_dice_loss:.6f}")
        print(f"val dice: {avg_dice:.6f}")
        print(f"val iou: {avg_iou:.6f}")
        print("per-class val metrics:")

        if self.class_names is None:
            max_cls = max(per_class_sum.keys()) if len(per_class_sum) > 0 else -1
            class_name_list = [str(i) for i in range(max_cls + 1)]
        else:
            class_name_list = self.class_names

        per_class_results = []
        for cls_id in sorted(per_class_sum.keys()):
            cnt = max(per_class_count[cls_id], 1)
            cls_dice = per_class_sum[cls_id]["dice"] / cnt
            cls_iou = per_class_sum[cls_id]["iou"] / cnt
            cls_name = class_name_list[cls_id] if cls_id < len(class_name_list) else str(cls_id)

            print(f"  [{cls_id}] {cls_name}: samples={cnt}, dice={cls_dice:.6f}, iou={cls_iou:.6f}")
            per_class_results.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "samples": cnt,
                "dice": cls_dice,
                "iou": cls_iou,
            })

            self.writer.add_scalar(f"val_per_class/class_{cls_id}_dice", cls_dice, epoch_idx)
            self.writer.add_scalar(f"val_per_class/class_{cls_id}_iou", cls_iou, epoch_idx)

        print("=====================================")

        self.writer.add_scalar("val/loss", avg_loss, epoch_idx)
        self.writer.add_scalar("val/bce", avg_bce, epoch_idx)
        self.writer.add_scalar("val/dice_loss", avg_dice_loss, epoch_idx)
        self.writer.add_scalar("val/dice", avg_dice, epoch_idx)
        self.writer.add_scalar("val/iou", avg_iou, epoch_idx)

        self.model.train()
        return {
            "epoch": epoch_idx,
            "loss": avg_loss,
            "bce": avg_bce,
            "dice_loss": avg_dice_loss,
            "dice": avg_dice,
            "iou": avg_iou,
            "per_class": per_class_results,
        }

    def save_checkpoint(self, epoch_idx):
        torch.save(
            {
                "epoch": epoch_idx,
                "decoder": self.model.decoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "class_names": self.class_names,
                "img_size": self.img_size,
                "alpha_mean": ALPHA_MEAN,
                "alpha_std": ALPHA_STD,
            },
            os.path.join(self.ckptdir, f"epoch_{epoch_idx}.pth"),
        )

    def load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.decoder.load_state_dict(ckpt["decoder"], strict=True)
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.class_names = ckpt.get("class_names", self.class_names)
        return int(ckpt.get("epoch", 0))

    def train(self, resume=False, amp=False):
        trainset = SonoULSegDataset(
            ids_file=self.train_txt,
            data_root=self.data_root,
            img_size=self.img_size,
            subnum=self.subnum,
        )
        testset = SonoULSegDataset(
            ids_file=self.test_txt,
            data_root=self.data_root,
            img_size=self.img_size,
            subnum=None,
        )

        if trainset.classes != testset.classes:
            raise RuntimeError(
                f"Train/Test classes mismatch:\ntrain={trainset.classes}\ntest={testset.classes}"
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
        test_loader = DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers_test,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        self.build_model_and_optimizer()

        total_steps = len(train_loader) * self.num_epoch
        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            base_lr=self.lr,
            warmup_steps=self.warmup_length,
            total_steps=total_steps,
        )

        start_epoch = 0
        global_step = 0

        if resume and os.path.isdir(self.ckptdir):
            ckpt_files = [f for f in os.listdir(self.ckptdir) if f.startswith("epoch_") and f.endswith(".pth")]
            if ckpt_files:
                ckpt_files = sorted(ckpt_files, key=lambda x: int(x[6:-4]))
                resume_pth = os.path.join(self.ckptdir, ckpt_files[-1])
                loaded_epoch = self.load_checkpoint(resume_pth)
                start_epoch = loaded_epoch
                global_step = loaded_epoch * len(train_loader)
                print(f"load resumed checkpoint: {resume_pth}")

        for epoch in range(start_epoch, self.num_epoch):
            self.model.train()

            epoch_loss_sum = 0.0
            epoch_bce_sum = 0.0
            epoch_dice_loss_sum = 0.0
            epoch_dice_sum = 0.0
            epoch_iou_sum = 0.0
            epoch_batch_count = 0

            for i, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch + 1}")):
                images, alpha_ones, masks, labels, stems, class_names_batch = batch

                step = global_step + i
                self.scheduler.step(step)

                images = images.to(self.device, non_blocking=True).float()
                alpha_ones = alpha_ones.to(self.device, non_blocking=True).float()
                masks = masks.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                use_amp = amp and self.device.type == "cuda"
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images, alpha_ones)
                        loss, loss_bce, loss_dice = self.compute_loss(logits, masks)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(images, alpha_ones)
                    loss, loss_bce, loss_dice = self.compute_loss(logits, masks)
                    loss.backward()
                    self.optimizer.step()

                batch_dice, batch_iou = batch_dice_iou(logits.detach(), masks)

                epoch_loss_sum += loss.item()
                epoch_bce_sum += loss_bce.item()
                epoch_dice_loss_sum += loss_dice.item()
                epoch_dice_sum += batch_dice
                epoch_iou_sum += batch_iou
                epoch_batch_count += 1

            global_step += len(train_loader)

            avg_train_loss = epoch_loss_sum / max(epoch_batch_count, 1)
            avg_train_bce = epoch_bce_sum / max(epoch_batch_count, 1)
            avg_train_dice_loss = epoch_dice_loss_sum / max(epoch_batch_count, 1)
            avg_train_dice = epoch_dice_sum / max(epoch_batch_count, 1)
            avg_train_iou = epoch_iou_sum / max(epoch_batch_count, 1)

            print("=====================================")
            print(f"epoch {epoch + 1}/{self.num_epoch} finished")
            print(f"train loss: {avg_train_loss:.6f}")
            print(f"train bce: {avg_train_bce:.6f}")
            print(f"train dice_loss: {avg_train_dice_loss:.6f}")
            print(f"train dice: {avg_train_dice:.6f}")
            print(f"train iou: {avg_train_iou:.6f}")
            print(f"lr: {self.optimizer.param_groups[0]['lr']:.8f}")
            print("=====================================")

            self.writer.add_scalar("train/epoch_loss", avg_train_loss, epoch + 1)
            self.writer.add_scalar("train/epoch_bce", avg_train_bce, epoch + 1)
            self.writer.add_scalar("train/epoch_dice_loss", avg_train_dice_loss, epoch + 1)
            self.writer.add_scalar("train/epoch_dice", avg_train_dice, epoch + 1)
            self.writer.add_scalar("train/epoch_iou", avg_train_iou, epoch + 1)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch + 1)

            # 最后一个 epoch 保存分割结果
            is_last_epoch = (epoch + 1 == self.num_epoch)

            # 每个 epoch 验证一次
            test_results = self.evaluate(
                test_loader,
                epoch + 1,
                save_results=is_last_epoch,
            )
            self._append_test_results(test_results)

            # 每个 epoch 保存一次
            self.save_checkpoint(epoch + 1)


# =========================
# Main
# =========================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="SonoCLIP frozen-backbone downstream segmentation (real alpha forward)")
    parser.add_argument(
        "--data_root",
        default=DEFAULT_SEG_DATA_ROOT,
        type=str,
        help="root containing class folders",
    )
    parser.add_argument(
        "--train_txt",
        default=os.path.join(DEFAULT_SEG_SPLIT_ROOT, "train.txt"),
        type=str,
        help="train split txt",
    )
    parser.add_argument(
        "--test_txt",
        default=os.path.join(DEFAULT_SEG_SPLIT_ROOT, "test.txt"),
        type=str,
        help="test split txt",
    )
    parser.add_argument("--base_model_path", default=DEFAULT_BASE_MODEL_PATH, type=str)
    parser.add_argument("--vision_ckpt", default=DEFAULT_VISION_CKPT_PATH, type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--exp_name", default="auto", type=str)
    parser.add_argument("--warmup_length", default=200, type=int)
    parser.add_argument("--epoch_num", default=20, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--subnum", default=None, type=int)
    parser.add_argument("--no_hi_res", action="store_true", help="use 224 instead of 336")
    parser.add_argument("--num_workers_train", default=8, type=int)
    parser.add_argument("--num_workers_test", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--bce_weight", default=0.5, type=float, help="final loss = bce_weight*BCE + (1-bce_weight)*Dice")
    parser.add_argument("--save_threshold", default=0.5, type=float, help="threshold for saving predicted binary mask")
    parser.add_argument("--decoder_mid_channels", default=512, type=int)
    args = parser.parse_args()

    trainer = SonoCLIPSegTrainer(
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        exp_name=args.exp_name,
        warmup_length=args.warmup_length,
        epoch_num=args.epoch_num,
        batch_size=args.batch_size,
        num_workers_train=args.num_workers_train,
        num_workers_test=args.num_workers_test,
        data_root=args.data_root,
        train_txt=args.train_txt,
        test_txt=args.test_txt,
        subnum=args.subnum,
        hi_res=(not args.no_hi_res),
        bce_weight=args.bce_weight,
        save_threshold=args.save_threshold,
        decoder_mid_channels=args.decoder_mid_channels,
        base_model_path=args.base_model_path,
        vision_ckpt=args.vision_ckpt,
    )
    trainer.train(resume=args.resume, amp=args.amp)
