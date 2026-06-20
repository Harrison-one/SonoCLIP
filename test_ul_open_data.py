import os
import json
import torch
import sonoclip
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import numpy as np
import cv2

# Define paths for model weights (参考 imagenet_s_zs_test.py)
SONOCLIP_MODEL = "ViT-L/14@336px"  # 基础模型名称
# 视觉编码器权重路径（训练后的视觉编码器权重）
SONOCLIP_VISUAL_CKPT = "/dat04/suhang/CLIP/pretrained/iter_55000.pth"  # 或使用预训练权重，如 "/dat04/suhang/CLIP/pretrained/clip_l14@336_grit_20m_4xe.pth"

# ============================================================================
# 图像类别名称配置 - 在这里插入你的图像类别
# ============================================================================
# 方式1: 直接在这里定义类别名称列表
# 注意：类别顺序应该与训练时一致（参考 plane_ids.txt）
# 训练时的顺序（从 plane_ids.txt）：
#   0: abdominal circumference
#   5: early-pregnancy transventricular
#   6: femur
#   8: four-chamber
#  20: transcerebellar
#  21: transthalamic
# 
# 重要：必须使用与训练时完全一致的类别名称（从 plane_ids.txt 中读取）
# 不要添加 "view" 等后缀，否则文本嵌入会不匹配
DEFAULT_CLASS_NAMES = [
    # 按照训练时的顺序（sorted 后的顺序），使用与训练时完全一致的名称
    "abdominal circumference",      # ID 0
    "cerebellar", 
    "early-pregnancy head",  # ID 5
    "femur",                      # ID 6
    "four-chamber", 
    "transthalamic",              # ID 21
]

# 方式2: 或者从文件读取（取消下面的注释并指定文件路径）
# DEFAULT_CLASS_NAMES_FILE = "/path/to/class_names.txt"
# ============================================================================

# Templates for zero-shot classification
# 注意：训练时只使用了一个模板 "A fetal ultrasound {} view."
# 但使用多个模板可以提高鲁棒性（对多个模板的嵌入取平均）
# 为了与训练时保持一致，第一个模板应该与训练时相同
simple_templates = [
    "A fetal ultrasound {} view.",  # 与训练时一致
    "An ultrasound image of the fetal {} view.",
    "A standard fetal ultrasound scan showing the {} view.",
    "This is a fetal ultrasound image in the {} view.",
    "A prenatal ultrasound image of the fetal {} view.",
]

# Image extensions to support
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")

# Mask transform (similar to imagenet_s.py)
mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336)),
    transforms.Normalize(0.5, 0.26)
])

# 自定义 preprocess，与训练时一致（使用 padding，不使用 center_crop）
# 参考 sono_ul_txt.py 的 res_clip_standard_transform
def create_padding_preprocess(n_px=336):
    """创建与训练时一致的预处理函数（padding + resize，不使用 center_crop）"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((n_px, n_px), interpolation=Image.BICUBIC),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class ImageFolderDataset(Dataset):
    """从指定文件夹读取图片的数据集
    支持两种模式：
    1. 单文件夹模式：image_folder 直接包含图片文件
    2. 子文件夹模式：image_folder 包含多个子文件夹，每个子文件夹名作为类别，子文件夹内包含该类别的图片
    
    对于 SonoCLIP，需要同时读取对应的 mask 文件（mask 数值为 255）
    """
    def __init__(self, image_folder, preprocess=None, class_names=None, use_subfolders=True, mask_folder=None, use_mask=True):
        """
        Args:
            image_folder: 图片文件夹路径
            preprocess: 图片预处理函数（如 preprocess）
            class_names: 类别名称列表（用于零样本分类，如果为None且use_subfolders=True，则从子文件夹名自动获取）
            use_subfolders: 是否使用子文件夹结构（子文件夹名作为类别）
            mask_folder: mask 文件夹路径，如果为None则从 image_folder 同级目录查找 mask 文件夹
            use_mask: 是否使用 mask 进行预测（True: 使用mask, False: 不使用mask，使用全1mask）
        """
        self.image_folder = os.path.abspath(image_folder)
        self.preprocess = preprocess
        self.use_subfolders = use_subfolders
        self.use_mask = use_mask  # 添加 use_mask 参数
        
        # 确定 mask 文件夹路径
        if not use_mask:
            # 如果不使用 mask，直接设为 None，强制使用全1 mask
            self.mask_folder = None
        elif mask_folder is None:
            # 默认在 image_folder 同级目录查找 mask 文件夹
            parent_dir = os.path.dirname(self.image_folder)
            potential_mask_folder = os.path.join(parent_dir, "masks")
            if os.path.exists(potential_mask_folder):
                self.mask_folder = potential_mask_folder
            else:
                # 或者在同一目录下
                potential_mask_folder = os.path.join(self.image_folder, "masks")
                if os.path.exists(potential_mask_folder):
                    self.mask_folder = potential_mask_folder
                else:
                    self.mask_folder = None
                    print("Warning: No mask folder found, will use all-ones masks")
        else:
            self.mask_folder = os.path.abspath(mask_folder)
        
        if use_subfolders:
            # 子文件夹模式：每个子文件夹名作为类别
            self.image_paths = []
            self.labels = []  # 存储类别ID
            self.class_to_idx = {}  # 类别名到ID的映射
            self.idx_to_class = {}  # ID到类别名的映射
            
            # 获取所有子文件夹
            subfolders = [f for f in os.listdir(self.image_folder) 
                         if os.path.isdir(os.path.join(self.image_folder, f))]
            subfolders = sorted(subfolders)
            
            if len(subfolders) == 0:
                raise ValueError(f"No subfolders found in {self.image_folder}")
            
            # 如果提供了class_names，使用提供的；否则从子文件夹名自动获取
            if class_names is not None:
                # 检查类别名称是否与子文件夹名匹配
                # 支持模糊匹配：如果子文件夹名包含类别名称，或者类别名称包含子文件夹名
                matched_subfolders = []
                unmatched_class_names = []
                folder_to_class = {}  # 子文件夹名 -> 类别名称的映射
                
                for class_name in class_names:
                    matched = False
                    for folder_name in subfolders:
                        # 精确匹配
                        if class_name == folder_name:
                            matched_subfolders.append(folder_name)
                            folder_to_class[folder_name] = class_name
                            matched = True
                            break
                        # 模糊匹配：检查是否包含（忽略大小写）
                        elif class_name.lower() in folder_name.lower() or folder_name.lower() in class_name.lower():
                            if folder_name not in folder_to_class:
                                matched_subfolders.append(folder_name)
                                folder_to_class[folder_name] = class_name
                                matched = True
                                break
                    
                    if not matched:
                        unmatched_class_names.append(class_name)
                
                if unmatched_class_names:
                    print(f"Warning: Some class names not found in subfolders: {unmatched_class_names}")
                    print(f"Available subfolders: {subfolders}")
                
                # 使用 class_names 作为类别列表（保持顺序），但使用匹配的子文件夹来读取图片
                self.classes = class_names  # 保持 class_names 的顺序
                self.folder_to_class = folder_to_class  # 保存映射关系
                subfolders = matched_subfolders
            else:
                self.classes = subfolders
            
            # 创建类别到ID的映射（按照 self.classes 的顺序）
            for idx, class_name in enumerate(self.classes):
                self.class_to_idx[class_name] = idx
                self.idx_to_class[idx] = class_name
            
            print(f"Class mapping (ID -> Name):")
            for idx, class_name in enumerate(self.classes):
                print(f"  {idx} -> {class_name}")
            
            # 收集每个子文件夹下的图片（只从 images 子文件夹中读取）
            # 注意：使用 folder_to_class 映射来找到对应的子文件夹
            if hasattr(self, 'folder_to_class') and self.folder_to_class:
                # 如果使用了 class_names，需要通过 folder_to_class 映射
                for folder_name, class_name in self.folder_to_class.items():
                    class_folder = os.path.join(self.image_folder, folder_name)
                    if not os.path.isdir(class_folder):
                        print(f"Warning: Subfolder {folder_name} not found, skipping...")
                        continue
                    
                    class_idx = self.class_to_idx[class_name]
                    
                    # 只从 {class_folder}/images/ 目录下收集图片
                    images_folder = os.path.join(class_folder, "images")
                    if not os.path.isdir(images_folder):
                        print(f"Warning: images folder not found in {class_folder}, skipping...")
                        continue
                    
                    # 收集 images 文件夹下的所有图片
                    class_images = []
                    for ext in IMAGE_EXTS:
                        class_images.extend(glob.glob(os.path.join(images_folder, f"*{ext}")))
                        class_images.extend(glob.glob(os.path.join(images_folder, f"*{ext.upper()}")))
                    
                    # 为每个图片添加对应的类别ID
                    self.image_paths.extend(class_images)
                    self.labels.extend([class_idx] * len(class_images))
            else:
                # 如果没有 folder_to_class 映射，直接使用类别名称作为文件夹名
                for class_name in self.classes:
                    # 如果类别名称不在匹配的 subfolders 中，跳过
                    if class_name not in subfolders:
                        print(f"Warning: Class '{class_name}' not found in subfolders, skipping...")
                        continue
                    class_folder = os.path.join(self.image_folder, class_name)
                    if not os.path.isdir(class_folder):
                        print(f"Warning: Subfolder {class_name} not found, skipping...")
                        continue
                    
                    class_idx = self.class_to_idx[class_name]
                    
                    # 只从 {class_folder}/images/ 目录下收集图片
                    images_folder = os.path.join(class_folder, "images")
                    if not os.path.isdir(images_folder):
                        print(f"Warning: images folder not found in {class_folder}, skipping...")
                        continue
                    
                    # 收集 images 文件夹下的所有图片
                    class_images = []
                    for ext in IMAGE_EXTS:
                        class_images.extend(glob.glob(os.path.join(images_folder, f"*{ext}")))
                        class_images.extend(glob.glob(os.path.join(images_folder, f"*{ext.upper()}")))
                    
                    # 为每个图片添加对应的类别ID
                    self.image_paths.extend(class_images)
                    self.labels.extend([class_idx] * len(class_images))
            
            # 移除重复的图片路径（如果有）
            unique_paths = []
            unique_labels = []
            seen = set()
            for path, label in zip(self.image_paths, self.labels):
                if path not in seen:
                    unique_paths.append(path)
                    unique_labels.append(label)
                    seen.add(path)
            
            self.image_paths = unique_paths
            self.labels = unique_labels
            
            print(f"Found {len(self.classes)} classes in subfolders:")
            for class_name in self.classes:
                class_count = sum(1 for l in self.labels if self.idx_to_class[l] == class_name)
                print(f"  {class_name}: {class_count} images")
            print(f"Total images: {len(self.image_paths)}")
            
        else:
            # 单文件夹模式：直接读取文件夹下的所有图片
            self.image_paths = []
            for ext in IMAGE_EXTS:
                self.image_paths.extend(glob.glob(os.path.join(self.image_folder, f"*{ext}")))
                self.image_paths.extend(glob.glob(os.path.join(self.image_folder, f"*{ext.upper()}")))
            
            self.image_paths = sorted(self.image_paths)
            self.labels = None  # 单文件夹模式没有标签
            
            if len(self.image_paths) == 0:
                raise ValueError(f"No images found in {self.image_folder}")
            
            # 类别名称（如果提供）
            self.classes = class_names if class_names is not None else []
            
            print(f"Found {len(self.image_paths)} images in {self.image_folder}")
            if self.classes:
                print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def _find_mask_path(self, image_path):
        """根据图片路径找到对应的 mask 文件
        mask_folder 的目录是在 images 同级下的 masks，每个切面都是
        例如：如果 image_path 是 /path/to/four-chamber/images/img.png
        那么 mask_path 应该是 /path/to/four-chamber/masks/img.png
        """
        # 如果不使用 mask，直接返回 None，强制使用全1 mask
        if not self.use_mask:
            return None
        
        # 获取图片文件名（不含扩展名）
        image_basename = os.path.basename(image_path)
        image_stem = os.path.splitext(image_basename)[0]
        
        # 如果指定了 mask_folder，直接使用
        if self.mask_folder is not None:
            # 在指定的 mask_folder 中查找
            for ext in IMAGE_EXTS:
                mask_path = os.path.join(self.mask_folder, image_stem + ext)
                if os.path.exists(mask_path):
                    return mask_path
                mask_path = os.path.join(self.mask_folder, image_stem + ext.upper())
                if os.path.exists(mask_path):
                    return mask_path
        
        # 如果子文件夹模式，在每个切面的 masks 文件夹中查找（与 images 同级）
        if self.use_subfolders:
            # 获取图片所在的切面文件夹路径
            # 例如：image_path = /path/to/four-chamber/images/img.png
            # 切面文件夹 = /path/to/four-chamber
            image_dir = os.path.dirname(image_path)
            # 获取切面文件夹（images 的父目录）
            plane_folder = os.path.dirname(image_dir)
            # 在该切面文件夹下查找 masks 文件夹
            plane_masks_folder = os.path.join(plane_folder, "masks")
            
            if os.path.exists(plane_masks_folder):
                for ext in IMAGE_EXTS:
                    mask_path = os.path.join(plane_masks_folder, image_stem + ext)
                    if os.path.exists(mask_path):
                        return mask_path
                    mask_path = os.path.join(plane_masks_folder, image_stem + ext.upper())
                    if os.path.exists(mask_path):
                        return mask_path
        
        return None
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        
        try:
            # 读取图片（与测试代码一致：使用 cv2 读取，然后转换为 RGB）
            # 这样可以确保与训练时的处理方式完全一致
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f"cannot read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 读取对应的 mask 文件（与训练时一致：先 padding，再 resize）
            mask_path = self._find_mask_path(image_path)
            if mask_path is not None and self.use_mask:
                # 读取 mask 文件
                mask = Image.open(mask_path)
                mask_array = np.array(mask)
                
                # 创建 binary_mask（bool 矩阵，参考示例代码）
                # get `binary_mask` array (2-dimensional bool matrix)
                if len(mask_array.shape) == 2:
                    binary_mask = (mask_array == 255)
                elif len(mask_array.shape) == 3:
                    binary_mask = (mask_array[:, :, 0] == 255)
                else:
                    raise ValueError(f"Unsupported mask shape: {mask_array.shape}")
                
                # 转换为 uint8（0 或 255），参考示例代码
                mask_uint8 = (binary_mask * 255).astype(np.uint8)
                
                # 重要：与测试代码一致，如果 mask 和 image 尺寸不匹配，旋转 image
                # 参考 ul_s_test.py 的处理方式
                if mask_uint8.shape != image.shape[:2]:
                    image = np.rot90(image)
                
                # 将 image 和 mask 合并为 rgba（与测试代码一致）
                rgba = np.concatenate((image, np.expand_dims(mask_uint8, axis=-1)), axis=-1)
            else:
                # 如果没有找到 mask 文件或 use_mask=False，使用全 1 的 mask
                mask_uint8 = np.ones(image.shape[:2], dtype=np.uint8) * 255
                rgba = np.concatenate((image, np.expand_dims(mask_uint8, axis=-1)), axis=-1)
            
            h, w = rgba.shape[:2]
            
            # Padding 到正方形（与训练时 choice="padding" 的处理一致）
            if max(h, w) == w:  # 横向图像
                pad = (w - h) // 2
                l, r = pad, w - h - pad
                rgba = np.pad(rgba, ((l, r), (0, 0), (0, 0)), "constant", constant_values=0)
            else:  # 纵向图像
                pad = (h - w) // 2
                l, r = pad, h - w - pad
                rgba = np.pad(rgba, ((0, 0), (l, r), (0, 0)), "constant", constant_values=0)
            
            # 分离 rgb 和 mask（与测试代码一致）
            rgb = rgba[:, :, :-1]
            mask = rgba[:, :, -1]
            
            # 转换为 PIL Image 以便使用 preprocess
            rgb_pil = Image.fromarray(rgb.astype(np.uint8))
            
            # 使用 preprocess 处理图片（应该只包含 Resize 和 Normalize，不包含 CenterCrop）
            if self.preprocess is not None:
                image_tensor = self.preprocess(rgb_pil)
            else:
                # 如果没有提供预处理函数，使用默认的 padding_preprocess
                image_tensor = create_padding_preprocess(336)(rgb_pil)
            
            # 处理 mask（与测试代码一致：mask * 255 然后 mask_transform）
            if self.use_mask and mask_path is not None:
                # 使用真实的 mask
                mask_tensor = mask_transform(Image.fromarray(mask, mode='L'))
            else:
                # 使用全 1 的 mask
                mask_tensor = mask_transform(Image.fromarray(np.ones_like(mask) * 255, mode='L'))
            
            # 如果有标签，返回图片、mask、标签和路径；否则返回图片、mask和路径
            if self.labels is not None:
                label = self.labels[index]
                return image_tensor, mask_tensor, label, image_path
            else:
                return image_tensor, mask_tensor, image_path
        
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise
def zeroshot_classifier(classnames, templates, model, device):
    """Build zero-shot classifier weights from class prompts.
    
    支持多个模板：为每个类别生成多个文本prompt，然后对它们的嵌入取平均。
    
    Args:
        classnames: 类别名称列表
        templates: 模板列表，例如 ["A fetal ultrasound {} view.", ...]
        model: SonoCLIP模型
        device: 设备
    
    Returns:
        zeroshot_weights: shape [embedding_dim, num_classes] 的零样本分类器权重
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Computing text embeddings"):
            # 为每个类别生成多个文本（每个模板一个）
            # 例如：["A fetal ultrasound four-chamber view.", "An ultrasound image of the fetal four-chamber view.", ...]
            texts = [template.format(classname) for template in templates]
            # 使用 sonoclip.tokenize
            texts = sonoclip.tokenize(texts).to(device)
            
            # 编码所有文本，得到 shape [num_templates, embedding_dim]
            class_embeddings = model.encode_text(texts)
            # 归一化每个模板的嵌入
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            
            # 对所有模板的嵌入取平均，得到该类别的综合嵌入
            class_embedding = class_embeddings.mean(dim=0)
            # 再次归一化
            class_embedding = class_embedding / class_embedding.norm()
            
            zeroshot_weights.append(class_embedding)
        
        # 堆叠所有类别的权重，shape: [embedding_dim, num_classes]
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    
    print(f"Generated zero-shot weights for {len(classnames)} classes using {len(templates)} templates per class")
    return zeroshot_weights


def evaluate_sonoclip(image_folder, class_names=None, use_subfolders=True, mask_folder=None, use_mask=True):
    """
    Evaluate SonoCLIP on images from a specified folder.
    
    Args:
        image_folder: 图片文件夹路径
        class_names: 类别名称列表（用于零样本分类），如果为None且use_subfolders=True，则从子文件夹名自动获取
        use_subfolders: 是否使用子文件夹结构（子文件夹名作为类别）
        mask_folder: mask 文件夹路径，如果为None则自动查找
        use_mask: 是否使用 mask 进行预测（True: 使用mask, False: 不使用mask，使用全1mask）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 显示 mask 接入状态
    print("=" * 50)
    print("Mask Support Status:")
    if use_mask:
        print("  ✓ SonoCLIP supports mask input")
        print(f"  Using mask from: {mask_folder if mask_folder else 'auto-detected (each plane folder/masks)'}")
        print("  Image encoding: model.visual(images, masks)")
    else:
        print("  ⚠ Mask input disabled (--no_mask)")
        print("  Using all-ones mask (full image)")
        print("  Image encoding: model.visual(images, all_ones_mask)")
    print("=" * 50)

    # Load SonoCLIP model (参考 imagenet_s_zs_test.py)
    # 使用 alpha_vision_ckpt_pth 参数加载视觉编码器权重
    # 如果 SONOCLIP_VISUAL_CKPT 是训练检查点（包含 visual, logit_scale, logit_bias），需要先提取 visual 部分
    if SONOCLIP_VISUAL_CKPT and os.path.exists(SONOCLIP_VISUAL_CKPT):
        # 检查是否是训练检查点格式（包含 "visual" 键）
        try:
            ckpt = torch.load(SONOCLIP_VISUAL_CKPT, map_location="cpu", weights_only=False)
            if "visual" in ckpt:
                # 这是训练检查点，需要提取 visual 部分并保存为临时文件，或者直接加载
                # 方法1：直接使用 sonoclip.load 加载基础模型，然后手动加载 visual
                print(f"Loading model with checkpoint from {SONOCLIP_VISUAL_CKPT}...")
                model, preprocess = sonoclip.load(SONOCLIP_MODEL, device=device)
                
                # 加载 visual 权重
                visual_state_dict = ckpt["visual"]
                # 处理 DDP 包装的情况
                if any(k.startswith("module.") for k in visual_state_dict.keys()):
                    visual_state_dict = {k.replace("module.", ""): v for k, v in visual_state_dict.items()}
                model.visual.load_state_dict(visual_state_dict, strict=False)
                
                # 加载 logit_scale 和 logit_bias（如果存在）
                if "logit_scale" in ckpt and hasattr(model, "logit_scale"):
                    model.logit_scale.data.copy_(ckpt["logit_scale"].to(device))
                if "logit_bias" in ckpt and hasattr(model, "logit_bias"):
                    model.logit_bias.data.copy_(ckpt["logit_bias"].to(device))
                
                print(f"Successfully loaded checkpoint from step {ckpt.get('step', 'unknown')}")
            else:
                # 这是直接的视觉编码器权重文件，可以直接使用 alpha_vision_ckpt_pth
                print(f"Loading model with vision checkpoint from {SONOCLIP_VISUAL_CKPT}...")
                model, preprocess = sonoclip.load(SONOCLIP_MODEL, alpha_vision_ckpt_pth=SONOCLIP_VISUAL_CKPT, device=device)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {SONOCLIP_VISUAL_CKPT}: {e}")
            print("Trying to load as direct vision checkpoint...")
            # 如果失败，尝试直接作为视觉编码器权重加载
            model, preprocess = sonoclip.load(SONOCLIP_MODEL, alpha_vision_ckpt_pth=SONOCLIP_VISUAL_CKPT, device=device)
    else:
        # 没有提供视觉编码器权重，只加载基础模型
        print(f"Loading base model {SONOCLIP_MODEL}...")
        model, preprocess = sonoclip.load(SONOCLIP_MODEL, device=device)
    
    model.eval()
    model = model.float()  # 确保模型是 float 类型（参考训练代码）

    # 创建与训练时一致的预处理函数（padding + resize，不使用 center_crop）
    # 参考 sono_ul_txt.py 的 res_clip_standard_transform
    # 不使用 sonoclip.load 返回的 preprocess（它包含 CenterCrop）
    padding_preprocess = create_padding_preprocess(336)
    
    print("Using padding-based preprocessing (consistent with training)")
    print("  Processing: Padding → Resize(336, 336) → Normalize")
    print("  (No center crop, preserving full image content)")

    # Load images from specified folder with padding-based preprocessing
    # 使用 padding_preprocess 处理图片（与训练时一致）
    # 如果class_names为None且use_subfolders=True，将从子文件夹名自动获取类别
    # 如果 use_mask=False，mask_folder 设为 None，这样数据集会生成全1的mask
    dataset = ImageFolderDataset(
        image_folder=image_folder,
        preprocess=padding_preprocess,  # 使用自定义的 padding_preprocess
        class_names=class_names,
        use_subfolders=use_subfolders,
        mask_folder=mask_folder if use_mask else None,
        use_mask=use_mask  # 明确传递 use_mask 参数
    )
    
    # 如果数据集自动获取了类别名称，更新class_names
    if class_names is None and hasattr(dataset, 'classes') and len(dataset.classes) > 0:
        class_names = dataset.classes
        print(f"Auto-detected classes from subfolders: {class_names}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=4,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # 如果有类别名称，生成零样本分类器权重
    if class_names and len(class_names) > 0:
        print("Generating zero-shot classifier weights...")
        print(f"Using {len(simple_templates)} templates per class")
        print(f"Class names for zero-shot: {class_names}")
        zeroshot_weights = zeroshot_classifier(
            class_names, simple_templates, model, device
        )
        
        # 调试：检查文本嵌入的相似度
        print("\nDebug: Checking text embedding similarities...")
        with torch.no_grad():
            # 计算类别之间的文本嵌入相似度
            for i, name1 in enumerate(class_names):
                texts1 = [t.format(name1) for t in simple_templates]
                texts1 = sonoclip.tokenize(texts1).to(device)
                emb1 = model.encode_text(texts1)
                emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
                emb1_mean = emb1.mean(dim=0) / emb1.mean(dim=0).norm()
                
                similarities = []
                for j, name2 in enumerate(class_names):
                    if i == j:
                        continue
                    texts2 = [t.format(name2) for t in simple_templates]
                    texts2 = sonoclip.tokenize(texts2).to(device)
                    emb2 = model.encode_text(texts2)
                    emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
                    emb2_mean = emb2.mean(dim=0) / emb2.mean(dim=0).norm()
                    
                    sim = (emb1_mean @ emb2_mean).item()
                    similarities.append((name2, sim))
                
                # 显示最相似的类别
                similarities.sort(key=lambda x: -x[1])
                print(f"  {name1} is most similar to: {similarities[0][0]} (sim={similarities[0][1]:.3f})")
        print()
    else:
        zeroshot_weights = None
        print("No class names provided, skipping classification.")

    print(f"Processing {len(dataset)} images from {image_folder}...")

    all_predictions = []
    all_image_paths = []
    all_true_labels = []  # 存储真实标签（如果有）

    with torch.no_grad():
        # 检查数据集是否返回标签
        sample = dataset[0]
        has_labels = len(sample) == 4  # (image, mask, label, path) 或 (image, mask, path)
        
        for batch in tqdm(loader, desc="Processing images"):
            if has_labels:
                images, masks, labels, image_paths = batch
                labels = labels.to(device, non_blocking=True)
                all_true_labels.extend(labels.cpu().numpy().tolist())
            else:
                if len(batch) == 3:
                    images, masks, image_paths = batch
                else:
                    images = batch[0]
                    masks = batch[1] if len(batch) > 1 else None
                    image_paths = [f"image_{i}" for i in range(len(images))]
            
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # SonoCLIP 使用 model.visual(images, masks) 进行图像编码（参考 train_ul_1m_sig_new.py）
            # 如果不使用 mask，masks 应该是全1的mask（已经在数据集中处理）
            image_features = model.visual(images, masks)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if zeroshot_weights is not None:
                # 零样本分类
                # 参考 train_ul_1m_sig_new.py 的 test_epoch，直接使用 torch.matmul 计算分数
                # 不使用 logit_scale，直接计算相似度
                score = torch.matmul(image_features, zeroshot_weights)

                pred = score.topk(1, dim=1)[1].squeeze(dim=1)
                k = min(5, score.size(1))
                pred_5 = score.topk(k, dim=1)[1]

                # 保存预测结果
                for idx in range(len(image_paths)):
                    pred_class_idx = pred[idx].item()
                    pred_class = class_names[pred_class_idx] if pred_class_idx < len(class_names) else "unknown"
                    top5_indices = pred_5[idx].tolist()
                    top5_classes = [class_names[i] if i < len(class_names) else "unknown" for i in top5_indices]
                    
                    pred_info = {
                        'image_path': image_paths[idx],
                        'predicted_class': pred_class,
                        'predicted_class_idx': pred_class_idx,
                        'top5_classes': top5_classes,
                        'scores': score[idx].cpu().numpy().tolist() if idx < score.size(0) else []
                    }
                    
                    # 如果有真实标签，添加准确率信息
                    if has_labels and idx < len(labels):
                        true_label = labels[idx].item()
                        true_class = dataset.idx_to_class.get(true_label, "unknown")
                        pred_info['true_class'] = true_class
                        pred_info['true_class_idx'] = true_label
                        pred_info['correct'] = (pred_class_idx == true_label)
                    
                    all_predictions.append(pred_info)
                    all_image_paths.append(image_paths[idx])
            else:
                # 只提取特征，不进行分类
                for idx in range(len(image_paths)):
                    pred_info = {
                        'image_path': image_paths[idx],
                        'features': image_features[idx].cpu().numpy().tolist()
                    }
                    if has_labels and idx < len(labels):
                        true_label = labels[idx].item()
                        true_class = dataset.idx_to_class.get(true_label, "unknown")
                        pred_info['true_class'] = true_class
                        pred_info['true_class_idx'] = true_label
                    all_predictions.append(pred_info)
                    all_image_paths.append(image_paths[idx])

    # 打印结果
    print("\n" + "=" * 50)
    if zeroshot_weights is not None:
        print("Classification Results:")
        print("=" * 50)
        
        # 如果有真实标签，计算准确率
        if all_true_labels:
            correct_count = sum(1 for p in all_predictions if p.get('correct', False))
            total_count = len(all_predictions)
            accuracy = correct_count / total_count * 100.0 if total_count > 0 else 0.0
            
            print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
            print()
            
            # 按类别统计准确率
            if hasattr(dataset, 'classes'):
                class_stats = {}
                for class_name in dataset.classes:
                    class_idx = dataset.class_to_idx[class_name]
                    class_samples = [p for p in all_predictions if p.get('true_class_idx') == class_idx]
                    if len(class_samples) > 0:
                        class_correct = sum(1 for p in class_samples if p.get('correct', False))
                        class_acc = class_correct / len(class_samples) * 100.0
                        class_stats[class_name] = (class_correct, len(class_samples), class_acc)
                
                print("Per-class Accuracy:")
                for class_name, (correct, total, acc) in sorted(class_stats.items()):
                    print(f"  {class_name}: {acc:.2f}% ({correct}/{total})")
                
                # 添加混淆矩阵信息：显示每个类别的预测分布
                print("\nPrediction Distribution (confusion matrix):")
                print("True Class -> Predicted Class counts:")
                confusion = {}
                for pred_info in all_predictions:
                    if 'true_class' in pred_info and 'predicted_class' in pred_info:
                        true_cls = pred_info['true_class']
                        pred_cls = pred_info['predicted_class']
                        if true_cls not in confusion:
                            confusion[true_cls] = {}
                        if pred_cls not in confusion[true_cls]:
                            confusion[true_cls][pred_cls] = 0
                        confusion[true_cls][pred_cls] += 1
                
                for true_cls in sorted(confusion.keys()):
                    print(f"  {true_cls}:")
                    for pred_cls, count in sorted(confusion[true_cls].items(), key=lambda x: -x[1])[:5]:
                        print(f"    -> {pred_cls}: {count}")
                print()
        
        # 显示前10个结果
        for pred_info in all_predictions[:10]:
            print(f"Image: {os.path.basename(pred_info['image_path'])}")
            if 'true_class' in pred_info:
                print(f"  True: {pred_info['true_class']}")
            print(f"  Predicted: {pred_info['predicted_class']}")
            print(f"  Top-5: {', '.join(pred_info['top5_classes'])}")
            if 'correct' in pred_info:
                status = "✓" if pred_info['correct'] else "✗"
                print(f"  {status}")
            print()
        if len(all_predictions) > 10:
            print(f"... and {len(all_predictions) - 10} more images")
    else:
        print(f"Extracted features for {len(all_predictions)} images")
    print("=" * 50)
    
    return all_predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate SonoCLIP on images from a folder')
    parser.add_argument(
        '--image_folder',
        type=str,
        default='/dat04/suhang/CLIP/test_planes_data/open_data/zero_shot_six_plane/test_cls_plane',
        help='图片文件夹路径'
    )
    parser.add_argument(
        '--class_names',
        type=str,
        nargs='+',
        default=None,
        help='类别名称列表（用于零样本分类），例如: --class_names "4CH" "Cereb" "Abdominal"。如果不指定，将使用代码中定义的 DEFAULT_CLASS_NAMES'
    )
    parser.add_argument(
        '--class_names_file',
        type=str,
        default=None,
        help='从文件读取类别名称（每行一个类别名）'
    )
    parser.add_argument(
        '--use_default_classes',
        action='store_true',
        help='使用代码中定义的 DEFAULT_CLASS_NAMES（如果已设置）'
    )
    parser.add_argument(
        '--use_subfolders',
        action='store_true',
        default=True,
        help='使用子文件夹结构（子文件夹名作为类别），默认启用'
    )
    parser.add_argument(
        '--no_subfolders',
        action='store_false',
        dest='use_subfolders',
        help='不使用子文件夹结构，直接从指定文件夹读取所有图片'
    )
    parser.add_argument(
        '--mask_folder',
        type=str,
        default=None,
        help='mask 文件夹路径，如果为None则自动查找（在 image_folder 同级目录或子目录中查找 masks 文件夹）'
    )
    parser.add_argument(
        '--use_mask',
        action='store_true',
        default=True,
        help='是否使用 mask 进行预测（默认True：使用mask，False：不使用mask，使用全1mask）'
    )
    parser.add_argument(
        '--no_mask',
        action='store_false',
        dest='use_mask',
        help='不使用 mask，使用全1的mask进行预测'
    )
    
    args = parser.parse_args()
    
    # 获取类别名称（优先级：命令行参数 > 文件 > 默认值 > None）
    if args.class_names_file:
        with open(args.class_names_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
        print(f"从文件读取类别名称: {args.class_names_file}")
        print(f"类别数量: {len(class_names)}")
    elif args.class_names:
        class_names = args.class_names
        print(f"使用命令行指定的类别名称，类别数量: {len(class_names)}")
    elif args.use_default_classes or (DEFAULT_CLASS_NAMES and len(DEFAULT_CLASS_NAMES) > 0):
        class_names = DEFAULT_CLASS_NAMES
        print(f"使用代码中定义的 DEFAULT_CLASS_NAMES，类别数量: {len(class_names)}")
        print(f"类别列表: {class_names}")
    else:
        class_names = None
        print("警告: 未指定类别名称，将只提取图像特征，不进行分类")
    
    # 执行评估
    evaluate_sonoclip(
        image_folder=args.image_folder,
        class_names=class_names,
        use_subfolders=args.use_subfolders,
        mask_folder=args.mask_folder,
        use_mask=args.use_mask
    )