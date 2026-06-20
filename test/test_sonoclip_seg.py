import os
import sys
import argparse
import cv2
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def _is_sonoclip_editable_finder(finder):
    return getattr(finder, "__module__", "").startswith("__editable___sonoclip_")


sys.meta_path[:] = [
    finder for finder in sys.meta_path
    if not _is_sonoclip_editable_finder(finder)
]

import sonoclip


# =========================
# Transforms
# =========================
PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)

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
        transforms.ToTensor(),
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


def _load_split_ids(ids_file):
    """
    test.txt:
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
    return items


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def require_file(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def require_dir(path, name):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def denorm_image_to_uint8(img_tensor_chw):
    """
    img_tensor_chw: [3,H,W], normalized
    return: uint8 RGB [H,W,3]
    """
    img = img_tensor_chw.detach().cpu().clone()
    mean = torch.tensor(PIXEL_MEAN, dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor(PIXEL_STD, dtype=img.dtype).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1)
    img = (img * 255.0).round().byte()
    img = img.permute(1, 2, 0).numpy()  # HWC RGB
    return img


def seg_test_collate_fn(batch):
    """
    固定尺寸输入 stack；原图相关信息保留为 list
    """
    images = torch.stack([x[0] for x in batch], dim=0)       # [B,3,S,S]
    alpha_ones = torch.stack([x[1] for x in batch], dim=0)   # [B,1,S,S]
    masks = torch.stack([x[2] for x in batch], dim=0)        # [B,1,S,S]
    labels = torch.tensor([x[3] for x in batch], dtype=torch.long)

    stems = [x[4] for x in batch]
    class_names_batch = [x[5] for x in batch]

    meta_list = [x[6] for x in batch]     # dict list
    orig_img_list = [x[7] for x in batch] # each: [H,W,3]
    orig_gt_list = [x[8] for x in batch]  # each: [H,W]

    return (
        images,
        alpha_ones,
        masks,
        labels,
        stems,
        class_names_batch,
        meta_list,
        orig_img_list,
        orig_gt_list,
    )


# =========================
# Dataset
# =========================
class SonoULSegTestDataset(Dataset):
    def __init__(self, ids_file, data_root, img_size=336):
        self.items = _load_split_ids(ids_file)
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

        return img, mask, image_path, mask_path

    def __getitem__(self, index):
        stem, label, class_name = self.items[index]
        img, mask, image_path, mask_path = self._read_image_and_mask(stem, class_name)

        # 对齐（和训练一致）
        rotated = False
        if mask.shape != img.shape[:2]:
            img = np.rot90(img)
            rotated = True
            if mask.shape != img.shape[:2]:
                raise RuntimeError(
                    f"image/mask shape mismatch after rotation: "
                    f"stem={stem}, img={img.shape[:2]}, mask={mask.shape}"
                )

        # 保存“对齐后的原图坐标系”
        orig_img = img.copy()   # RGB, HxWx3
        orig_gt = mask.copy()   # Gray, HxW
        orig_h, orig_w = orig_gt.shape[:2]

        # 记录 pad 信息（逆变换要用）
        h, w = img.shape[:2]
        pad_top = pad_bottom = pad_left = pad_right = 0

        # pad to square（和训练一致）
        if h < w:
            pad = (w - h) // 2
            pad_top, pad_bottom = pad, w - h - pad
            img = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="constant", constant_values=0)
            mask = np.pad(mask, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0)
        elif w < h:
            pad = (h - w) // 2
            pad_left, pad_right = pad, h - w - pad
            img = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=0)
            mask = np.pad(mask, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)

        side = img.shape[0]  # pad 后正方形边长（也等于 img.shape[1]）

        image_torch = self.img_transform(img)          # [3,S,S]
        mask_torch = self.mask_transform(mask).float() # [1,S,S]
        mask_torch = (mask_torch > 0.5).float()

        # real alpha forward: 全1 alpha + 按 SonoCLIP 预期归一化
        alpha_ones = torch.ones_like(mask_torch)
        alpha_ones = (alpha_ones - ALPHA_MEAN) / ALPHA_STD

        meta = {
            "orig_h": int(orig_h),
            "orig_w": int(orig_w),
            "side": int(side),
            "pad_top": int(pad_top),
            "pad_bottom": int(pad_bottom),
            "pad_left": int(pad_left),
            "pad_right": int(pad_right),
            "rotated": bool(rotated),
            "image_path": image_path,
            "mask_path": mask_path,
        }

        orig_img_t = torch.from_numpy(orig_img.copy())  # [H,W,3], uint8
        orig_gt_t = torch.from_numpy(orig_gt.copy())    # [H,W], uint8

        return (
            image_torch,
            alpha_ones,
            mask_torch,
            label,
            stem,
            class_name,
            meta,
            orig_img_t,
            orig_gt_t,
        )


# =========================
# Frozen SonoCLIP ViT dense feature extractor
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
                "Current SonoCLIP visual does not expose conv1 / conv1_alpha."
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
        v = self.visual

        if alpha is None:
            raise ValueError("SonoCLIP visual forward assumes alpha is not None.")

        images = images.to(dtype=v.conv1.weight.dtype)
        alpha = alpha.to(dtype=v.conv1_alpha.weight.dtype)

        if alpha.shape[-2:] != images.shape[-2:]:
            alpha = F.interpolate(
                alpha,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = v.conv1(images) + v.conv1_alpha(alpha)   # [B,C,Gh,Gw]
        if hasattr(v, "dwconv"):
            x = v.dwconv(x)
        B, C, Gh, Gw = x.shape

        x = x.reshape(B, C, Gh * Gw).permute(0, 2, 1)  # [B,N,C]

        cls = v.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)  # [B,1+N,C]

        pos = v.positional_embedding.to(x.dtype)
        if pos.shape[0] != x.shape[1]:
            raise RuntimeError(
                f"Positional embedding length mismatch: pos={pos.shape[0]}, tokens={x.shape[1]}"
            )
        x = x + pos

        x = v.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = v.transformer(x)
        x = x.permute(1, 0, 2)

        x = x[:, 1:, :]
        feat_map = x.permute(0, 2, 1).reshape(B, C, Gh, Gw)
        return feat_map


# =========================
# Decoder
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
# Build model
# =========================
def build_model(
    device="cuda",
    hi_res=True,
    base_model_path="",
    visual_ckpt="",
    decoder_ckpt="",
    decoder_mid_channels=512,
):
    require_file(base_model_path, "base ViT checkpoint")
    require_file(visual_ckpt, "SonoCLIP visual checkpoint")
    require_file(decoder_ckpt, "decoder checkpoint")

    base_model, _ = sonoclip.load(
        base_model_path,
        device="cpu",
        lora_adapt=False,
        rank=-1,
    )

    ckpt = torch.load(visual_ckpt, map_location="cpu", weights_only=False)
    if "visual" in ckpt:
        visual_state_dict = ckpt["visual"]
        if any(k.startswith("module.") for k in visual_state_dict.keys()):
            visual_state_dict = {k.replace("module.", ""): v for k, v in visual_state_dict.items()}
        msg = base_model.visual.load_state_dict(visual_state_dict, strict=False)
        print(f"Loaded visual encoder from: {visual_ckpt}")
        print(f"Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
    else:
        msg = base_model.visual.load_state_dict(ckpt, strict=False)
        print(f"Loaded visual weights from: {visual_ckpt}")
        print(f"Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")

    img_size = 336 if hi_res else 224
    model = SonoCLIPFrozenSegModel(
        base_model.float(),
        img_size=img_size,
        decoder_mid_channels=decoder_mid_channels,
    )

    dec_ckpt = torch.load(decoder_ckpt, map_location="cpu", weights_only=False)
    if "decoder" not in dec_ckpt:
        raise KeyError(f"decoder checkpoint does not contain key 'decoder': {decoder_ckpt}")
    model.decoder.load_state_dict(dec_ckpt["decoder"], strict=True)
    print(f"Loaded decoder from: {decoder_ckpt}")

    model = model.to(device)
    model.eval()
    return model, img_size


# =========================
# 逆变换：把 336/224 的预测正确还原回原图坐标
# =========================
def restore_pred_to_original(pred_small, meta):
    """
    pred_small: [S,S] uint8, 0/255
    meta: 包含 side / pad_top / pad_bottom / pad_left / pad_right / orig_h / orig_w

    正确流程：
    1) 先从 SxS resize 回 pad 后的正方形 side x side
    2) 再裁掉 padding
    3) 得到与对齐后的原图一致的尺寸
    """
    side = int(meta["side"])
    pad_top = int(meta["pad_top"])
    pad_bottom = int(meta["pad_bottom"])
    pad_left = int(meta["pad_left"])
    pad_right = int(meta["pad_right"])
    orig_h = int(meta["orig_h"])
    orig_w = int(meta["orig_w"])

    # step1: 回到 pad 后正方形
    pred_pad = cv2.resize(pred_small, (side, side), interpolation=cv2.INTER_NEAREST)

    # step2: 裁掉 pad
    y0 = pad_top
    y1 = side - pad_bottom
    x0 = pad_left
    x1 = side - pad_right
    pred_crop = pred_pad[y0:y1, x0:x1]

    # 安全兜底（理论上这里已经等于 orig_h, orig_w）
    if pred_crop.shape[0] != orig_h or pred_crop.shape[1] != orig_w:
        pred_crop = cv2.resize(pred_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    pred_crop = (pred_crop > 127).astype(np.uint8) * 255
    return pred_crop


# =========================
# Dice & Save helper
# =========================
def dice_single(pred_mask_uint8, gt_mask_uint8, eps=1e-6):
    """单张二值 mask 的 Dice，pred/gt 为 0/255 的 uint8。"""
    p = (pred_mask_uint8 > 127).astype(np.float32).flatten()
    g = (gt_mask_uint8 > 127).astype(np.float32).flatten()
    inter = (p * g).sum()
    return (2.0 * inter + eps) / (p.sum() + g.sum() + eps)


def iou_single(pred_mask_uint8, gt_mask_uint8, eps=1e-6):
    """单张二值 mask 的 IoU，pred/gt 为 0/255 的 uint8。"""
    p = (pred_mask_uint8 > 127).astype(np.float32).flatten()
    g = (gt_mask_uint8 > 127).astype(np.float32).flatten()
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return (inter + eps) / (union + eps)


def save_restore_masks(output_root, class_name, stem, orig_gt, restore_pred):
    """只保存恢复到原图空间的 GT mask 和预测 mask。"""
    dirs = {
        "gt_mask_restore": os.path.join(output_root, "gt_mask_restore", class_name),
        "pred_mask_restore": os.path.join(output_root, "pred_mask_restore", class_name),
    }
    for d in dirs.values():
        ensure_dir(d)

    cv2.imwrite(os.path.join(dirs["gt_mask_restore"], f"{stem}_gt.png"), orig_gt)
    cv2.imwrite(os.path.join(dirs["pred_mask_restore"], f"{stem}_pred.png"), restore_pred)


@torch.no_grad()
def predict_test_set(args=None):
    if args is None:
        args = parse_args()

    require_dir(args.data_root, "data root")
    require_file(args.test_txt, "test split txt")

    model, img_size = build_model(
        device=args.device,
        hi_res=(not args.no_hi_res),
        base_model_path=args.base_model,
        visual_ckpt=args.visual_ckpt,
        decoder_ckpt=args.decoder_ckpt,
        decoder_mid_channels=args.decoder_mid_channels,
    )

    dataset = SonoULSegTestDataset(
        ids_file=args.test_txt,
        data_root=args.data_root,
        img_size=img_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
        collate_fn=seg_test_collate_fn,
    )

    ensure_dir(args.output_dir)
    total = 0
    metrics_by_class = {}  # class_name -> {count, dice_sum, iou_sum}
    dice_sum_all = 0.0
    iou_sum_all = 0.0

    for batch in tqdm(loader, desc="Predicting test set"):
        (
            images,
            alpha_ones,
            masks,
            labels,
            stems,
            class_names_batch,
            meta_list,
            orig_img_list,
            orig_gt_list,
        ) = batch

        images = images.to(args.device, non_blocking=True).float()
        alpha_ones = alpha_ones.to(args.device, non_blocking=True).float()

        logits = model(images, alpha_ones)
        probs = torch.sigmoid(logits)
        preds = (probs > args.threshold).float()   # [B,1,S,S]

        preds_np = preds.detach().cpu().numpy()

        bs = images.size(0)
        for b in range(bs):
            stem = stems[b]
            class_name = class_names_batch[b]
            meta = meta_list[b]

            input_pred = (preds_np[b, 0] > 0.5).astype(np.uint8) * 255         # [S,S]

            # 原图空间（对齐后的原图）
            orig_gt = (orig_gt_list[b].numpy() > 127).astype(np.uint8) * 255   # [H,W]

            # 正确逆变换恢复
            restore_pred = restore_pred_to_original(input_pred, meta)

            save_restore_masks(
                output_root=args.output_dir,
                class_name=class_name,
                stem=stem,
                orig_gt=orig_gt,
                restore_pred=restore_pred,
            )

            dice_val = dice_single(restore_pred, orig_gt)
            iou_val = iou_single(restore_pred, orig_gt)
            if class_name not in metrics_by_class:
                metrics_by_class[class_name] = {
                    "count": 0,
                    "dice_sum": 0.0,
                    "iou_sum": 0.0,
                }
            metrics_by_class[class_name]["count"] += 1
            metrics_by_class[class_name]["dice_sum"] += float(dice_val)
            metrics_by_class[class_name]["iou_sum"] += float(iou_val)
            dice_sum_all += float(dice_val)
            iou_sum_all += float(iou_val)

            total += 1

    per_class_metrics = []
    for class_name in sorted(metrics_by_class.keys()):
        rec = metrics_by_class[class_name]
        count = rec["count"]
        mean_dice = rec["dice_sum"] / max(count, 1)
        mean_iou = rec["iou_sum"] / max(count, 1)
        per_class_metrics.append((class_name, count, mean_dice, mean_iou))

    mean_dice_all = dice_sum_all / max(total, 1)
    mean_iou_all = iou_sum_all / max(total, 1)
    macro_dice = sum(x[2] for x in per_class_metrics) / max(len(per_class_metrics), 1)
    macro_iou = sum(x[3] for x in per_class_metrics) / max(len(per_class_metrics), 1)

    metrics_txt = os.path.join(args.output_dir, "metrics_summary.txt")
    metrics_csv = os.path.join(args.output_dir, "metrics_per_class.csv")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"total_samples={total}\n")
        f.write(f"mean_dice_sample={mean_dice_all:.6f}\n")
        f.write(f"mean_iou_sample={mean_iou_all:.6f}\n")
        f.write(f"mean_dice_class_macro={macro_dice:.6f}\n")
        f.write(f"mean_iou_class_macro={macro_iou:.6f}\n")
    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write("class_name,count,mean_dice,mean_iou\n")
        for class_name, count, mean_dice, mean_iou in per_class_metrics:
            f.write(f"{class_name},{count},{mean_dice:.6f},{mean_iou:.6f}\n")


    print("-" * 60)
    print("Final metrics:")
    print(f"  mean Dice over samples: {mean_dice_all:.4f}")
    print(f"  mean IoU  over samples: {mean_iou_all:.4f}")
    print(f"  mean Dice over classes: {macro_dice:.4f}")
    print(f"  mean IoU  over classes: {macro_iou:.4f}")
    print("-" * 60)
    print("Per-class mean metrics:")
    for class_name, count, mean_dice, mean_iou in per_class_metrics:
        print(f"  {class_name}: n={count}  dice={mean_dice:.4f}  iou={mean_iou:.4f}")
    print("-" * 60)
    print(f"Done. Saved {total} samples to: {args.output_dir}")
    print("Saved outputs:")
    print(f"  {args.output_dir}/gt_mask_restore")
    print(f"  {args.output_dir}/pred_mask_restore")
    print(f"  {metrics_txt}")
    print(f"  {metrics_csv}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict SonoCLIP segmentation masks.")
    parser.add_argument("--data-root", required=True, help="Test segmentation data root.")
    parser.add_argument("--test-txt", required=True, help="Test split txt.")
    parser.add_argument("--base-model", required=True, help="Path to the base ViT/CLIP checkpoint.")
    parser.add_argument("--visual-ckpt", required=True, help="Path to the SonoCLIP visual checkpoint.")
    parser.add_argument("--decoder-ckpt", required=True, help="Path to the decoder checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--best-per-class-dir",
        default=None,
        help="Deprecated; accepted for compatibility but no best-sample outputs are saved.",
    )
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--decoder-mid-channels", default=512, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-hi-res", action="store_true", help="Use 224px input instead of 336px.")
    args = parser.parse_args()

    args.data_root = os.path.abspath(args.data_root)
    args.test_txt = os.path.abspath(args.test_txt)
    args.base_model = os.path.abspath(args.base_model)
    args.visual_ckpt = os.path.abspath(args.visual_ckpt)
    args.decoder_ckpt = os.path.abspath(args.decoder_ckpt)
    args.output_dir = os.path.abspath(args.output_dir)
    return args


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    predict_test_set()
