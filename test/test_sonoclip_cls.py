import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _is_sonoclip_editable_finder(finder):
    return getattr(finder, "__module__", "").startswith("__editable___sonoclip_")


sys.meta_path[:] = [
    finder for finder in sys.meta_path
    if not _is_sonoclip_editable_finder(finder)
]

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)


def _find_file_by_stem(folder, stem):
    for ext in IMG_EXTS:
        p = os.path.join(folder, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def _load_split_ids(ids_file):
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

def _image_transform(hi_res=True):
    side = 336 if hi_res else 224
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((side, side), interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
    ])


def _mask_transform(hi_res=True):
    side = 336 if hi_res else 224
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((side, side), interpolation=InterpolationMode.NEAREST),
        transforms.Normalize(0.5, 0.26),
    ])


class SonoClsDataset(Dataset):
    def __init__(self, ids_file, data_root, hi_res=True, use_mask=False):
        self.items = _load_split_ids(ids_file)
        self.data_root = os.path.abspath(data_root)
        self.use_mask = bool(use_mask)
        self.image_transform = _image_transform(hi_res=hi_res)
        self.mask_transform = _mask_transform(hi_res=hi_res)

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
        return img, image_path

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
            img = np.pad(img, ((l, r), (0, 0), (0, 0)), mode="constant", constant_values=0)
            if mask is not None:
                mask = np.pad(mask, ((l, r), (0, 0)), mode="constant", constant_values=0)
        else:
            pad = (h - w) // 2
            l, r = pad, h - w - pad
            img = np.pad(img, ((0, 0), (l, r), (0, 0)), mode="constant", constant_values=0)
            if mask is not None:
                mask = np.pad(mask, ((0, 0), (l, r)), mode="constant", constant_values=0)
        return img, mask

    def __getitem__(self, index):
        stem, label, class_name = self.items[index]
        img, _ = self._read_image(stem, class_name)

        if self.use_mask:
            mask = self._read_mask(stem, class_name)
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

        image_torch = self.image_transform(img)
        mask_torch = self.mask_transform(mask)

        return index, image_torch, mask_torch, label


def _load_model(model, ckpt_path, device):
    if not ckpt_path:
        return {"loaded": False, "path": None, "message": "no checkpoint specified"}
    if not os.path.isfile(ckpt_path):
        return {"loaded": False, "path": ckpt_path, "message": "checkpoint not found"}

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["visual"] if isinstance(ckpt, dict) and "visual" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    load_msg = model.visual.load_state_dict(state_dict, strict=False)
    model.to(device)
    return {
        "loaded": True,
        "path": ckpt_path,
        "step": int(ckpt.get("step", -1)) if isinstance(ckpt, dict) else -1,
        "missing_keys": len(load_msg.missing_keys),
        "unexpected_keys": len(load_msg.unexpected_keys),
    }


def _build_cls_head(state_dict, in_dim, num_classes, device):
    if "0.weight" in state_dict and "3.weight" in state_dict:
        hidden_dim = int(state_dict["0.weight"].shape[0])
        head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
        head.load_state_dict(state_dict, strict=True)
        return head.to(device), f"mlp_hidden={hidden_dim}"

    if "weight" in state_dict and "bias" in state_dict:
        head = nn.Linear(in_dim, num_classes)
        head.load_state_dict(state_dict, strict=True)
        return head.to(device), "linear"

    raise ValueError(f"cannot infer classifier head type from keys: {list(state_dict.keys())[:8]}")


def _compute_acc_macro_f1(y_true, y_pred, labels):
    return {
        "accuracy_top1": float(accuracy_score(y_true, y_pred) * 100.0),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) * 100.0
        ),
    }


def _save_per_class_csv(path, rows):
    fieldnames = [
        "class_id",
        "class_name",
        "total",
        "correct",
        "acc",
        "precision",
        "recall",
        "f1",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_summary_metrics_csv(path, metrics):
    rows = [
        {
            "metric": "accuracy_top1",
            "value": metrics["accuracy_top1"],
        },
        {
            "metric": "macro_f1",
            "value": metrics["macro_f1"],
        },
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _save_confusion_matrix_png(path, confmat, class_names):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to save confusion_matrix.png") from exc

    fig_w = max(8.0, 0.45 * len(class_names))
    fig_h = max(6.0, 0.45 * len(class_names))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(confmat, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = list(range(len(class_names)))
    ax.set(
        xticks=labels,
        yticks=labels,
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = confmat.max() / 2.0 if confmat.size else 0.0
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            val = int(confmat[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _build_per_class_rows(confmat, class_names):
    labels = list(range(len(class_names)))
    y_true = []
    y_pred = []
    for true_id in labels:
        for pred_id in labels:
            count = int(confmat[true_id, pred_id])
            if count > 0:
                y_true.extend([true_id] * count)
                y_pred.extend([pred_id] * count)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    rows = []
    for class_id, class_name in enumerate(class_names):
        total = int(support[class_id])
        correct = int(confmat[class_id, class_id])
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "total": total,
            "correct": correct,
            "acc": float(correct / total * 100.0) if total > 0 else 0.0,
            "precision": float(precision[class_id] * 100.0),
            "recall": float(recall[class_id] * 100.0),
            "f1": float(f1[class_id] * 100.0),
        })
    return rows


@torch.no_grad()
def evaluate(model, cls_head, loader, class_names, device):
    model.eval()
    cls_head.eval()

    all_targets = []
    all_preds = []
    num_classes = len(class_names)
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for _, images, masks, labels in tqdm(loader, desc="Testing"):
        images = images.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        feats = model.visual(images, masks).float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        logits = cls_head(feats)

        pred1 = logits.topk(1, dim=1)[1].squeeze(1)

        for i in range(labels.shape[0]):
            pred_id = int(pred1[i].item())
            target_id = int(labels[i].item())

            confmat[target_id, pred_id] += 1
            all_targets.append(target_id)
            all_preds.append(pred_id)

    return all_targets, all_preds, confmat


def main(args=None):
    if args is None:
        args = parse_args()

    checkpoint = args.classifier_ckpt
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    hi_res = not args.no_hi_res
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    import sonoclip

    model_name = "ViT-L/14@336px" if hi_res else "ViT-L/14"
    if not os.path.isfile(args.base_model):
        raise FileNotFoundError(f"base model not found: {args.base_model}")
    model, _ = sonoclip.load(args.base_model, device="cpu", lora_adapt=False, rank=-1)
    visual_load_info = _load_model(model, args.visual_ckpt, device)
    model = model.float().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    dataset = SonoClsDataset(
        ids_file=args.test_txt,
        data_root=args.data_root,
        hi_res=hi_res,
        use_mask=args.use_mask,
    )
    class_names = ckpt.get("class_names", dataset.classes)
    if class_names != dataset.classes:
        print(f"Warning: checkpoint classes differ from test set classes:\nckpt={class_names}\ntest={dataset.classes}")

    num_classes = int(ckpt.get("num_classes", len(class_names)))
    in_dim = int(model.text_projection.shape[1])
    cls_head, head_type = _build_cls_head(
        ckpt["cls_head"],
        in_dim=in_dim,
        num_classes=num_classes,
        device=device,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    all_targets, all_preds, confmat = evaluate(
        model=model,
        cls_head=cls_head,
        loader=loader,
        class_names=class_names,
        device=device,
    )

    labels = sorted(set(int(x) for x in all_targets))
    metrics = _compute_acc_macro_f1(all_targets, all_preds, labels=labels)
    per_class_rows = _build_per_class_rows(confmat, class_names)

    per_class_csv = os.path.join(output_dir, "per_class_metrics.csv")
    summary_metrics_csv = os.path.join(output_dir, "summary_metrics.csv")
    confmat_png = os.path.join(output_dir, "confusion_matrix.png")

    _save_per_class_csv(per_class_csv, per_class_rows)
    _save_summary_metrics_csv(summary_metrics_csv, metrics)
    _save_confusion_matrix_png(confmat_png, confmat, class_names)

    print("\n" + "=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print(f"SonoCLIP visual ckpt: {args.visual_ckpt}")
    print(f"use_mask: {args.use_mask}")
    print("-" * 60)
    print("Final metrics:")
    print(f"accuracy/top1: {metrics['accuracy_top1']:.2f}%")
    print(f"macro_f1: {metrics['macro_f1']:.2f}%")
    print("-" * 60)
    print("Per-class metrics:")
    for row in per_class_rows:
        print(
            f"  {row['class_name']}: n={row['total']}  "
            f"acc={row['acc']:.2f}%  f1={row['f1']:.2f}%"
        )
    print("-" * 60)
    print(f"Saved per-class metrics: {per_class_csv}")
    print(f"Saved summary metrics: {summary_metrics_csv}")
    print(f"Saved confusion matrix: {confmat_png}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test SonoCLIP linear probe with accuracy and macro-F1."
    )
    parser.add_argument("--base-model", required=True, help="Path to the base ViT/CLIP checkpoint.")
    parser.add_argument("--visual-ckpt", required=True, help="Path to the SonoCLIP visual checkpoint.")
    parser.add_argument("--classifier-ckpt", required=True, help="Path to the classifier checkpoint.")
    parser.add_argument("--data-root", required=True, help="Test dataset root.")
    parser.add_argument("--test-txt", required=True, help="Test split txt.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and confusion matrix outputs.")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-mask", action="store_true", help="Use real masks at test time. Defaults to all-one masks.")
    parser.add_argument("--no-hi-res", action="store_true", help="Use ViT-L/14 224px instead of ViT-L/14@336px.")
    args = parser.parse_args()

    args.base_model = os.path.abspath(args.base_model)
    args.visual_ckpt = os.path.abspath(args.visual_ckpt)
    args.classifier_ckpt = os.path.abspath(args.classifier_ckpt)
    args.data_root = os.path.abspath(args.data_root)
    args.test_txt = os.path.abspath(args.test_txt)
    args.output_dir = os.path.abspath(args.output_dir)
    return args


if __name__ == "__main__":
    main()
