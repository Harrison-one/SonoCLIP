"""
Plane 分类数据集：参考 Imagenet_S，读 combined json 或每张图一个 json，
固定随机选用一个 mask 作为提示，每个 plane 对应一个类别 ID。
返回格式与 Imagenet_S 一致：image_torch, mask_torch, plane_id。
"""

import json
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils
from PIL import Image
from torchvision import transforms

clip_standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

res_clip_standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])

res_mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336)),
    transforms.Normalize(0.5, 0.26)
])

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_TEST_ROOT = os.path.join(PROJECT_ROOT, "ul_data", "test")
MASK_RANDOM_SEED = 2026


def _find_image_path(root_dir, stem):
    for ext in IMAGE_EXTS:
        p = os.path.join(root_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def _expand_ann_files(ann_files):
    expanded = []
    for path in ann_files:
        path = os.path.abspath(path)
        if os.path.isdir(path):
            expanded.extend(
                os.path.join(path, name)
                for name in sorted(os.listdir(path))
                if name.lower().endswith(".json")
            )
        else:
            expanded.append(path)
    return expanded


def _is_annotation_dict(data):
    return isinstance(data, dict) and (
        "ref_exps" in data or "seudo_masks" in data or "plane" in data
    )


def _iter_annotation_items(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_stem = os.path.splitext(os.path.basename(json_path))[0]
    if _is_annotation_dict(data):
        yield file_stem, data
        return

    if isinstance(data, dict):
        for stem, ann in data.items():
            if _is_annotation_dict(ann):
                yield stem, ann
        return

    if isinstance(data, list):
        for idx, ann in enumerate(data):
            if not _is_annotation_dict(ann):
                continue
            stem = ann.get("stem") or ann.get("image_id") or ann.get("file_name") or f"{file_stem}_{idx}"
            stem = os.path.splitext(os.path.basename(str(stem)))[0]
            yield stem, ann
        return

    raise TypeError(f"unsupported json format in {json_path}: {type(data)}")


def crop_center(img, croph, cropw):
    h, w = img.shape[:2]
    starth = h // 2 - (croph // 2)
    startw = w // 2 - (cropw // 2)
    return img[starth : starth + croph, startw : startw + cropw, :]


class UL_Test(Dataset):
    """
    读 combined json 或每张图一个 json，每个 plane 一个类别 ID，固定随机选一个 mask。
    - ann_files: 一个或多个 json 路径，或 json 目录。
      * combined json: {stem: ann, ...}
      * per-image json: 单个 ann dict，stem 使用 json 文件名
      * 类别统一从 ann["plane"] 字段收集，sorted 后分配 id
    - image_root: 图片根目录，文件名 {stem}.jpg/.png
    - choice: "center_crop" 或 "padding"
    - hi_res / all_one: 同 Imagenet_S
    - plane_ids_txt: 若传入路径，会写入「id 与 plane 对应」的 txt，每行 "id plane_name"
    """

    def __init__(
        self,
        ann_files=os.path.join(DEFAULT_TEST_ROOT, "ul_test.json"),
        image_root=os.path.join(DEFAULT_TEST_ROOT, "images"),
        choice="center_crop",
        hi_res=False,
        all_one=False,
        plane_ids_txt=os.path.join(DEFAULT_TEST_ROOT, "plane_ids.txt"),
    ):
        if isinstance(ann_files, str):
            ann_files = [ann_files]
        ann_files = _expand_ann_files(ann_files)
        image_root = os.path.abspath(image_root)

        self.image_root = image_root
        self.choice = choice
        self.all_one = all_one

        self.plane_names = []
        self.anns = []

        raw_anns = []
        for jpath in ann_files:
            for stem, ann in _iter_annotation_items(jpath):
                if not ann.get("ref_exps") or not ann.get("seudo_masks"):
                    continue
                raw_anns.append((stem, ann))

        planes_set = {ann.get("plane", "unknown") for _, ann in raw_anns}
        self.plane_names = sorted(planes_set) if planes_set else ["unknown"]
        plane_to_id = {p: i for i, p in enumerate(self.plane_names)}
        for stem, ann in raw_anns:
            plane_id = plane_to_id.get(ann.get("plane", "unknown"), 0)
            self.anns.append((stem, ann, plane_id))

        self.num_classes = len(self.plane_names)
        # 与 Imagenet_S 一致，供 zeroshot_classifier(test_loader.dataset.classes, ...) 使用
        self.classes = self.plane_names

        # 可选：生成 id 与 plane 对应的 txt
        if plane_ids_txt:
            plane_ids_txt = os.path.abspath(plane_ids_txt)
            os.makedirs(os.path.dirname(plane_ids_txt) or ".", exist_ok=True)
            with open(plane_ids_txt, "w", encoding="utf-8") as f:
                f.write("# id\tplane_name\n")
                for i, name in enumerate(self.plane_names):
                    f.write(f"{i}\t{name}\n")
            print("[OK] wrote plane id mapping:", plane_ids_txt)

        if hi_res:
            self.mask_transform = res_mask_transform
            self.clip_standard_transform = res_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        stem, ann, plane_id = self.anns[index]
        masks = ann["seudo_masks"]
        rng = np.random.default_rng(MASK_RANDOM_SEED + index)
        mask_rle = masks[int(rng.integers(0, len(masks)))]

        image_path = _find_image_path(self.image_root, stem)
        if image_path is None:
            raise FileNotFoundError(f"image not found for stem '{stem}' under {self.image_root}")
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = maskUtils.decode(mask_rle)
        if mask.shape != image.shape[:2]:
            image = np.rot90(image)

        rgba = np.concatenate((image, np.expand_dims(mask, axis=-1)), axis=-1)
        h, w = rgba.shape[:2]

        if self.choice == "padding":
            if max(h, w) == w:
                pad = (w - h) // 2
                l, r = pad, w - h - pad
                rgba = np.pad(rgba, ((l, r), (0, 0), (0, 0)), "constant", constant_values=0)
            else:
                pad = (h - w) // 2
                l, r = pad, h - w - pad
                rgba = np.pad(rgba, ((0, 0), (l, r), (0, 0)), "constant", constant_values=0)
        else:
            if min(h, w) == h:
                rgba = crop_center(rgba, h, h)
            else:
                rgba = crop_center(rgba, w, w)

        rgb = rgba[:, :, :-1]
        mask = rgba[:, :, -1]
        image_torch = self.clip_standard_transform(rgb)

        if self.all_one:
            mask_torch = self.mask_transform(np.ones_like(mask) * 255)
        else:
            mask_torch = self.mask_transform(mask * 255)

        return image_torch, mask_torch, plane_id


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=str, nargs="+", required=True, help="combined json path(s), one per plane or single multi-plane json")
    parser.add_argument("--image-root", type=str, required=True, help="image directory")
    parser.add_argument("--choice", type=str, default="center_crop", choices=["center_crop", "padding"])
    parser.add_argument("--hi-res", action="store_true")
    parser.add_argument("--plane-ids-txt", type=str, default=None, help="write id->plane mapping to this txt (e.g. plane_ids.txt)")
    args = parser.parse_args()

    ds = UL_Test(
        ann_files=args.ann,
        image_root=args.image_root,
        choice=args.choice,
        hi_res=args.hi_res,
        plane_ids_txt=args.plane_ids_txt,
    )
    print("plane_id -> name:", dict(enumerate(ds.plane_names)))
    print("num_classes:", ds.num_classes, "len:", len(ds))
    img, mask, pid = ds[0]
    print("sample 0: image", img.shape, "mask", mask.shape, "plane_id", pid)
