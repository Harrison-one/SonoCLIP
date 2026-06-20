"""
Sono_UL 数据集：用随机排序的 txt 作为 id 列表。
- ids_file: 支持 .txt（每行一个图片名前缀/stem）或 .pkl
- json 与 image 各一个对应，通过 stem 作为键；可指定 json 目录与图片目录分离
"""

import json
import os
import random
import pickle
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils
from PIL import Image
from torchvision import transforms

PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

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

def _load_ids(ids_file, subnum=None):
    """从 .txt（每行一个 stem）或 .pkl 加载 id 列表。"""
    ids_file = os.path.abspath(ids_file)
    if not os.path.isfile(ids_file):
        raise FileNotFoundError(f"ids_file not found: {ids_file}")

    if ids_file.endswith(".txt"):
        with open(ids_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        with open(ids_file, "rb") as f:
            ids = pickle.load(f)
        if not isinstance(ids, (list, tuple)):
            ids = list(ids)

    if subnum is not None:
        ids = ids[:subnum]
    return ids


def _find_image_path(root_dir, stem, exts):
    for ext in exts:
        image_path = os.path.join(root_dir, stem + ext)
        if os.path.isfile(image_path):
            return image_path
    return None


def crop_center(img, croph, cropw):
    h, w = img.shape[:2]
    starth = h // 2 - (croph // 2)
    startw = w // 2 - (cropw // 2)
    return img[starth : starth + croph, startw : startw + cropw, :]


class Sono_UL_Txt(Dataset):
    """
    使用 txt 或 pkl 作为 id 列表，从本地目录读取 json 与 image（一一对应，键为 stem）。
    - ids_file: .txt 每行一个 stem（图片名前缀），或 .pkl
    - json_root: 存放 {stem}.json 的目录
    - image_root: 存放图片的目录，优先读取 {stem}.png，读不到再尝试 jpg/jpeg/bmp/webp
    - 若只传 root_pth，则 json 与 image 都从 root_pth 下读。
    """

    def __init__(
        self,
        ids_file,
        root_pth=None,
        json_root=None,
        image_root=None,
        common_pair=0.0,
        hi_res=False,
        subnum=None,
    ):
        self.ids = _load_ids(ids_file, subnum=subnum)
        root_pth = root_pth or ""
        self.json_root = os.path.abspath(json_root if json_root is not None else root_pth)
        self.image_root = os.path.abspath(image_root if image_root is not None else root_pth)
        self.with_common_pair_prop = common_pair

        if hi_res:
            self.mask_transform = res_mask_transform
            self.clip_standard_transform = res_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform

    def __len__(self):
        return len(self.ids)

    def _read_ann_and_image(self, stem):
        json_path = os.path.join(self.json_root, stem + ".json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"json not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        image_path = os.path.join(self.image_root, stem + ".png")
        img = cv2.imread(image_path)
        if img is None:
            image_path = _find_image_path(self.image_root, stem, IMAGE_EXTS[1:])
            img = cv2.imread(image_path) if image_path is not None else None
        if img is None:
            raise RuntimeError(f"cannot read image for stem '{stem}' under {self.image_root}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ann, img

    def __getitem__(self, index):
        stem = self.ids[index]
        ann, img = self._read_ann_and_image(stem)

        use_common_pair = random.random() < self.with_common_pair_prop
        ref_exps = ann["ref_exps"]
        choice = random.randint(0, len(ref_exps) - 1)
        ref_exp = ref_exps[choice]
        # 若有 mask_captions：用选中的那一整句；否则用 caption 的 [start:end] 切片
        if "mask_captions" in ann and isinstance(ann["mask_captions"], (list, tuple)):
            text = ann["mask_captions"][choice]
        else:
            text = ann["caption"][int(ref_exp[0]) : int(ref_exp[1])]

        if use_common_pair:
            mask = np.ones(img.shape[:2], dtype=np.uint8)
        else:
            mask = maskUtils.decode(ann["seudo_masks"][choice])

        if mask.shape != img.shape[:2]:
            img = np.rot90(img)

        rgba = np.concatenate((img, np.expand_dims(mask, axis=-1)), axis=-1)
        h, w = rgba.shape[:2]
        choice = 0
        if choice == 0:
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

        if not use_common_pair:
            mask_torch = self.mask_transform(mask * 255)
            return image_torch, mask_torch, text
            
        mask_torch = self.mask_transform(mask * 255)
        return image_torch, mask_torch, ann["caption"]
            
        
