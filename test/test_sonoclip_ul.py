import os
import sys
import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_outputs", "ul_fea_per_image")
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def _is_sonoclip_editable_finder(finder):
    return getattr(finder, "__module__", "").startswith("__editable___sonoclip_")


sys.meta_path[:] = [
    finder for finder in sys.meta_path
    if not _is_sonoclip_editable_finder(finder)
]

import torch

from test_sonoclip_ul import simple_templates, zeroshot_classifier, save_metrics
from test_sonoclip_ul_fea import load_text_model, require_file


def require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("per-image h5 feature input requires h5py. Install it with: pip install h5py") from exc
    return h5py


def read_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return read_string(value.item())
        if value.size == 1:
            return read_string(value.reshape(-1)[0])
    return str(value)


def list_h5_files(features_dir):
    path = Path(features_dir)
    if not path.is_dir():
        raise NotADirectoryError(f"features dir not found: {features_dir}")
    files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in (".h5", ".hdf5")
    )
    if not files:
        raise FileNotFoundError(f"no .h5 files found in: {features_dir}")
    return files


def load_class_names_txt(path):
    if not path:
        return None
    require_file(path, "Class names txt")
    indexed = {}
    plain = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                indexed[int(parts[0])] = parts[1].strip()
            else:
                plain.append(line)

    if indexed:
        max_id = max(indexed)
        missing = [idx for idx in range(max_id + 1) if idx not in indexed]
        if missing:
            raise ValueError(f"class names txt has missing ids: {missing[:10]}")
        return [indexed[idx] for idx in range(max_id + 1)]
    return plain


def load_per_image_features(features_dir, class_names_txt=None):
    h5py = require_h5py()
    files = list_h5_files(features_dir)

    features = []
    labels = []
    label_to_name = {}
    feature_dim = None
    for path in files:
        with h5py.File(path, "r") as f:
            for key in ("image_features", "label", "class_name"):
                if key not in f:
                    raise KeyError(f"feature file missing key '{key}': {path}")

            feat = f["image_features"][:].astype(np.float32)
            if feat.ndim != 1:
                raise ValueError(f"image_features must be 1D [D], got {feat.shape}: {path}")
            if feature_dim is None:
                feature_dim = feat.shape[0]
            elif feat.shape[0] != feature_dim:
                raise ValueError(f"feature dim mismatch in {path}: {feat.shape[0]} != {feature_dim}")

            label = int(np.asarray(f["label"][()]).item())
            class_name = read_string(f["class_name"][()])
            old_name = label_to_name.get(label)
            if old_name is not None and old_name != class_name:
                raise ValueError(f"label {label} maps to both '{old_name}' and '{class_name}'")

            label_to_name[label] = class_name
            features.append(feat)
            labels.append(label)

    labels = np.asarray(labels, dtype=np.int64)
    class_names = load_class_names_txt(class_names_txt)
    if class_names is None:
        max_label = int(labels.max())
        missing = [idx for idx in range(max_label + 1) if idx not in label_to_name]
        if missing:
            raise ValueError(
                "per-image h5 files do not contain every class id up to max label. "
                f"Missing ids: {missing[:10]}. Provide --class-names-txt if this is expected."
            )
        class_names = [label_to_name[idx] for idx in range(max_label + 1)]
    else:
        max_label = int(labels.max())
        if max_label >= len(class_names):
            raise ValueError(
                f"label id {max_label} is out of range for class_names length {len(class_names)}"
            )

    image_features = torch.from_numpy(np.stack(features, axis=0)).float()
    labels = torch.from_numpy(labels).long()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return image_features, labels, class_names, files


def evaluate_per_image_features(args=None):
    if args is None:
        args = parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    image_features, labels, class_names, files = load_per_image_features(
        args.features_dir,
        class_names_txt=args.class_names_txt,
    )

    if args.vision_ckpt:
        require_file(args.vision_ckpt, "Vision checkpoint")
        print(f"Vision checkpoint argument: {args.vision_ckpt}")
        print("Note: feature-only evaluation uses pre-extracted image_features; --vision-ckpt is recorded only.")

    model = load_text_model(args.base_model, device)
    zeroshot_weights = zeroshot_classifier(class_names, simple_templates, model, device)
    zeroshot_weights = zeroshot_weights.to(dtype=image_features.dtype)

    temp_corr_dict = {}
    n = image_features.shape[0]
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            feats = image_features[start:end].to(device, non_blocking=True)
            target = labels[start:end].to(device, non_blocking=True)

            score = 100.0 * (feats @ zeroshot_weights)
            pred = score.topk(1, dim=1)[1].squeeze(dim=1)
            k = min(5, score.size(1))
            pred_5 = score.topk(k, dim=1)[1]

            for idx in range(target.shape[0]):
                cls = int(target[idx].item())
                if cls not in temp_corr_dict:
                    temp_corr_dict[cls] = [0, 0, 0]
                temp_corr_dict[cls][0] += 1
                if cls == int(pred[idx].item()):
                    temp_corr_dict[cls][1] += 1
                if cls in pred_5[idx].tolist():
                    temp_corr_dict[cls][2] += 1

    acc1 = 0.0
    acc5 = 0.0
    num_class = 0
    for v in temp_corr_dict.values():
        if v[0] == 0:
            continue
        acc1 += v[1] / v[0]
        acc5 += v[2] / v[0]
        num_class += 1

    if num_class == 0:
        raise RuntimeError("No valid samples found in feature directory.")

    acc1 = acc1 / num_class * 100.0
    acc5 = acc5 / num_class * 100.0

    print("\n" + "-" * 60)
    print("Per-class TOP1 / TOP5 results:")
    print(f"{'Class':<30} {'Total':>6} {'TOP1':>8} {'TOP5':>8} {'TOP1%':>8} {'TOP5%':>8}")
    print("-" * 60)
    per_class_rows = []
    for cls in sorted(temp_corr_dict.keys(), key=lambda c: class_names[c] if c < len(class_names) else c):
        v = temp_corr_dict[cls]
        if v[0] == 0:
            continue
        cls_name = class_names[cls] if cls < len(class_names) else str(cls)
        top1_pct = v[1] / v[0] * 100.0
        top5_pct = v[2] / v[0] * 100.0
        per_class_rows.append({
            "class_name": cls_name,
            "total": v[0],
            "top1": v[1],
            "top5": v[2],
            "top1_acc": top1_pct,
            "top5_acc": top5_pct,
        })
        print(f"{cls_name:<30} {v[0]:>6} {v[1]:>8} {v[2]:>8} {top1_pct:>7.2f}% {top5_pct:>7.2f}%")

    print("-" * 60)
    print(f">>> TOP1 accuracy (mean per-class): {acc1:.2f}%")
    print(f">>> TOP5 accuracy (mean per-class): {acc5:.2f}%")

    per_class_csv, summary_txt = save_metrics(args.output_dir, per_class_rows, acc1, acc5)
    print(f"Saved per-class metrics: {per_class_csv}")
    print(f"Saved summary metrics: {summary_txt}")
    print(f"Feature dir: {args.features_dir}")
    print(f"num_feature_files: {len(files)}")
    print(f"num_samples: {labels.numel()}")
    print(f"num_classes: {len(class_names)}")
    if args.vision_ckpt:
        print(f"Vision checkpoint argument: {args.vision_ckpt}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SonoCLIP UL zero-shot from anonymous per-image h5 features.")
    parser.add_argument("--base-model", required=True, help="Path to the base ViT/CLIP checkpoint for text encoder.")
    parser.add_argument("--vision-ckpt", default=None, help="Optional vision checkpoint path to record with this evaluation.")
    parser.add_argument("--features-dir", required=True, help="Directory containing anonymous per-image h5 files.")
    parser.add_argument("--class-names-txt", default=None, help="Optional id-to-class txt, e.g. plane_ids.txt.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for saved metric files.")
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    args.features_dir = os.path.abspath(args.features_dir)
    args.base_model = os.path.abspath(args.base_model)
    args.vision_ckpt = os.path.abspath(args.vision_ckpt) if args.vision_ckpt else None
    args.class_names_txt = os.path.abspath(args.class_names_txt) if args.class_names_txt else None
    args.output_dir = os.path.abspath(args.output_dir)
    return args


if __name__ == "__main__":
    evaluate_per_image_features()
