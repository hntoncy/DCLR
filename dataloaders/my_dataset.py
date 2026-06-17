import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from albumentations import ElasticTransform
from torchvision.transforms import functional as F


"""
/media/dell/codes/Dqw/Paper/contrast/SSL4MIS/data/ISIC2017/
/home/dell/codes/Dqw/code/SSL4MIS/data/ISIC2017/
├── Images/
│   ├── ISIC_0000000.jpg
│   └── ...
├── Labels/
│   ├── ISIC_0000000.jpg
│   └── ...
├── train.txt    # 训练集列表（1935 samples）
└── val.txt      # 验证集列表（215 samples）
"""

class BaseDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",  # "train" 或 "val"
            num=None,  # 限制训练样本数量（调试用）
            transform=None,
            ops_weak=None,  # 弱增强策略
            ops_strong=None  # 强增强策略
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        # 参数校验
        assert split in ("train", "val"), "split 必须是 'train' 或 'val'"
        assert bool(ops_weak) == bool(ops_strong), \
            "使用CTAugment时需要同时提供弱增强和强增强策略"

        # 加载样本列表文件
        list_file = os.path.join(base_dir, "train.txt" if split == "train" else "val.txt")
        with open(list_file, "r") as f:
            self.sample_list = [line.strip() for line in f.readlines()]

        # 限制训练样本数量
        if num is not None and split == "train":
            self.sample_list = self.sample_list[:num]
        print(f"Total  {len(self.sample_list)} {self.split} samples")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        """从JPG文件加载数据"""
        case = self.sample_list[idx]  # 如 "ISIC_0000000"

        # 构建图像和标签路径
        image_path = os.path.join(self._base_dir, "Images", f"{case}.png")  # 改1
        label_path = os.path.join(self._base_dir, "Labels", f"{case}.png")

        # 加载图像和标签
        # image = np.array(Image.open(image_path).convert("RGB"))  # 形状 (H, W, 3)
        # 修改为单通道（灰度）
        image = np.array(Image.open(image_path).convert("L"))  # 形状 (H, W)
        # image = np.expand_dims(image, axis=-1)  # 形状 (H, W, 1)，根据需要选择
        label = np.array(Image.open(label_path).convert("L"))  # 单通道灰度图

        # 归一化和预处理
        image = image.astype(np.float32) / 255.0  # 归一化到 [0,1]
        # label = (label > 128).astype(np.uint8)  # 二值化标签（假设标签是0-255掩码）
        label = np.clip(label, 0, 255)  # 防止极端像素值
        label = (label > 128).astype(np.float32)  # 先转浮点确保精度
        label = np.round(label).astype(np.uint8)  # 明确取整

        sample = {"image": image, "label": label}

        # 新增cutmix，以 50% 概率对训练集样本进行 CutMix 增强（混合另一个随机样本的图像块和标签块）
        if self.split == "train" and random.random() < 0.5:
            # 随机选择另一个样本
            idx2 = random.randint(0, len(self.sample_list) - 1)
            case2 = self.sample_list[idx2]
            image_path2 = os.path.join(self._base_dir, "Images", f"{case2}.png")  # 改2
            label_path2 = os.path.join(self._base_dir, "Labels", f"{case2}.png")

            image2 = np.array(Image.open(image_path2).convert("L")) / 255.0
            label2 = np.array(Image.open(label_path2).convert("L"))
            label2 = (label2 > 128).astype(np.uint8)

            image, label = cutmix(image, label, image2, label2)

        # 训练时应用数据增强
        if self.split == "train":
            if self.ops_weak is not None and self.ops_strong is not None:
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

        sample["idx"] = idx

        assert np.all(np.isin(label, [0, 1])), f"发现非法标签值: {np.unique(label)}"
        return sample

def cutmix(image1, label1, image2, label2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    if image1.ndim == 2:  # 灰度
        h, w = image1.shape
    elif image1.ndim == 3:  # RGB
        h, w, c = image1.shape
    cx, cy = np.random.randint(h), np.random.randint(w)
    cut_h = int(np.sqrt(lam) * h)
    cut_w = int(np.sqrt(lam) * w)
    x1 = max(cx - cut_h // 2, 0)
    y1 = max(cy - cut_w // 2, 0)
    x2 = min(cx + cut_h // 2, h)
    y2 = min(cy + cut_w // 2, w)

    image1[x1:x2, y1:y2] = image2[x1:x2, y1:y2]
    label1[x1:x2, y1:y2] = label2[x1:x2, y1:y2]
    return image1, label1

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)  # 随机旋转0,90,180,270度
        label = np.flip(label, axis=axis).copy()  # 随机沿水平轴或垂直轴翻转
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)  # order=0，改1
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

        self.elastic = ElasticTransform(
            p=0.5,  # 概率
            alpha=120,  # 变形强度
            sigma=120 * 0.05,  # 平滑程度
            border_mode=0
        )

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        rand_value = random.random()
        # 弹性形变，50%概率
        if random.random() < 0.5:
            augmented = self.elastic(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        if rand_value < 0.5:
            image, label = random_rot_flip(image, label)  # 随机旋转+翻转
        else:
            image, label = random_rotate(image, label)  # 随机角度旋转

        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # order=0，改2
        # label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label,
            (self.output_size[0] / x, self.output_size[1] / y),
            order=0,  # 强制使用最近邻插值
            mode="nearest"
        )
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class RandomGenerator_Aug(object):
    def __init__(self, output_size, noise_std=0.05, brightness=0.2, contrast=0.2):
        """
        Args:
            output_size: 目标尺寸 [H, W]
            noise_std: 高斯噪声的标准差 (相对于像素值范围 0-1)
            brightness: 亮度调整幅度
            contrast: 对比度调整幅度
        """
        self.output_size = output_size
        self.noise_std = noise_std
        self.brightness = brightness
        self.contrast = contrast

        # 初始化弹性形变
        self.elastic = ElasticTransform(
            p=0.5,
            alpha=120,
            sigma=120 * 0.05,
            border_mode=0
        )

    def _apply_color_jitter(self, image):
        """
        对单通道图像应用亮度和对比度调整
        """
        # 1. 亮度调整: pixel * (1 + delta)
        if self.brightness > 0:
            delta_b = random.uniform(-self.brightness, self.brightness)
            image = image * (1 + delta_b)

        # 2. 对比度调整: (pixel - mean) * (1 + delta) + mean
        if self.contrast > 0:
            delta_c = random.uniform(-self.contrast, self.contrast)
            mean_val = np.mean(image)
            image = (image - mean_val) * (1 + delta_c) + mean_val

        return image

    def _apply_gaussian_noise(self, image):
        """
        添加高斯噪声
        """
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
            image = image + noise
        return image

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # --- 步骤 1: 几何增强 (Elastic & Rotation/Flip) ---
        # 注意：几何增强应在颜色增强之前或之后均可，但通常先做几何再做颜色，
        # 或者先做颜色再做几何。这里保持原有顺序，先几何后颜色，或者根据需求调整。
        # 为了防止插值产生的边界伪影被颜色增强放大，建议先做几何增强。

        rand_value = random.random()

        # 1.1 弹性形变 (50% 概率)
        if random.random() < 0.5:
            augmented = self.elastic(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        # 1.2 随机旋转翻转 或 随机旋转
        if rand_value < 0.5:
            image, label = random_rot_flip(image, label)
        else:
            image, label = random_rotate(image, label)

        # --- 步骤 2: Resize ---
        x, y = image.shape
        # 图像使用双线性插值 (order=1) 以获得更好质量，标签使用最近邻 (order=0)
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # 改
        label = zoom(
            label,
            (self.output_size[0] / x, self.output_size[1] / y),
            order=0,
            mode="nearest"
        )

        # --- 步骤 3: 颜色/强度增强 (Color Jitter & Noise) ---
        # 注意：此时 image 仍然是 numpy array, 范围大约在 [0, 1]

        # 3.1 应用亮度和对比度调整
        image = self._apply_color_jitter(image)

        # 3.2 应用高斯噪声
        image = self._apply_gaussian_noise(image)

        # 3.3 截断到 [0, 1] 范围，防止溢出
        image = np.clip(image, 0, 1)

        # --- 步骤 4: 转换为 Tensor ---
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {"image": image, "label": label}
        return sample


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    # 在每个 Epoch 内对有标签索引进行一次随机排列（不重复）
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    # 无限循环地随机打乱无标签索引
    # 因为无标签数据通常远多于有标签数据，且需要在每个 Step 都填充 Batch，所以需要永恒迭代
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

# 新增 Diffrect使用
class c_random_flip:
    def __init__(self, lr=True, ud=True):
        self.lr = lr
        self.ud = ud

    def __call__(self, sample):
        lr = np.random.random() < 0.5 and self.lr is True
        ud = np.random.random() < 0.5 and self.ud is True
        # lr = self.lr
        # ud = self.ud

        for key in sample.keys():
            if key in ['image', 'image_weak', 'image_strong', 'gt', 'contour']:
                sample[key] = np.array(sample[key])
                if lr:
                    sample[key] = np.fliplr(sample[key])
                if ud:
                    sample[key] = np.flipud(sample[key])
                sample[key] = Image.fromarray(sample[key])

        return sample


class c_random_rotate:
    def __init__(self, range=[0, 360], interval=1):
        self.range = range
        self.interval = interval

    def __call__(self, sample):
        rot = (np.random.randint(*self.range) // self.interval) * self.interval
        rot = rot + 360 if rot < 0 else rot

        if np.random.random() < 0.5:
            for key in sample.keys():
                if key in ['image', 'image_weak', 'image_strong', 'gt', 'contour']:
                    base_size = sample[key].size

                    sample[key] = sample[key].rotate(rot, expand=True)

                    sample[key] = sample[key].crop(((sample[key].size[0] - base_size[0]) // 2,
                                                    (sample[key].size[1] - base_size[1]) // 2,
                                                    (sample[key].size[0] + base_size[0]) // 2,
                                                    (sample[key].size[1] + base_size[1]) // 2))

        return sample


class c_normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image_weak, image_strong = sample['image_weak'], sample['image_strong']
        if image.max() > 1:
            image /= 255
            image_weak /= 255
            image_strong /= 255
        else:
            # print('image max value is {:.3f}'.format(np.max(image)))
            pass
        if self.mean is not None and self.std is not None:
            image -= self.mean
            image /= self.std

            image_weak -= self.mean
            image_weak /= self.std

            image_strong -= self.mean
            image_strong /= self.std

        # norm to [0, 1] if max value is 255
        if np.max(gt) == 255:
            gt /= 255

        sample['image'] = image
        sample['gt'] = gt
        sample['image_weak'] = image_weak
        sample['image_strong'] = image_strong

        return sample


class c_random_image_enhance:
    def __init__(self, methods=['contrast', 'brightness', 'sharpness']):
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def __call__(self, sample):
        image = sample['image_strong']
        np.random.shuffle(self.enhance_method)

        for method in self.enhance_method:
            if np.random.random() > 0.5:
                enhancer = method(image)
                factor = float(1 + np.random.random() / 10)
                image = enhancer.enhance(factor)
        sample['image_strong'] = image

        return sample


class c_random_gaussian_blur:
    def __init__(self, apply=False):
        self.apply = apply

    def __call__(self, sample):
        if self.apply:
            image = sample['image_strong']
            if np.random.random() < 0.5:
                image = image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
            sample['image_strong'] = image

        return sample


class c_tonumpy:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image_weak, image_strong = sample['image_weak'], sample['image_strong']

        sample['image'] = np.array(image, dtype=np.float32)
        sample['gt'] = np.array(gt, dtype=np.float32)
        sample['image_weak'] = np.array(image_weak, dtype=np.float32)
        sample['image_strong'] = np.array(image_strong, dtype=np.float32)

        return sample

class WeakStrongAugment_Ours(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size, args=None, split='train'):
        self.output_size = output_size
        self.split = split

        self.transform_list_train = {
            'c_random_flip': {'lr': True, 'ud': True},
            'c_random_rotate': {'range': [0, 359]},
            'c_random_image_enhance': {'methods': ['contrast', 'sharpness', 'brightness']},
            'c_random_gaussian_blur': {'apply': True},
            'c_tonumpy': None,
            'c_normalize': {'mean': None, 'std': None}, }
        self.transform_list_test = {
            'c_tonumpy': None,
            'c_normalize': {'mean': None, 'std': None}, }

        if args:
            if args.no_color:
                self.transform_list_train.pop('c_random_image_enhance', None)
            if args.no_blur:
                self.transform_list_train.pop('c_random_gaussian_blur', None)
            if args.rot != 359:
                self.transform_list_train['c_random_rotate']['range'] = [-args.rot, args.rot]
            # if

        self.transform_list_train = self.get_transform(self.transform_list_train)
        self.transform_list_test = self.get_transform(self.transform_list_test)

    @staticmethod
    def get_transform(transform_list):

        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        image_untrans_np = image
        image = torch.from_numpy(image.astype(np.float32))
        image = transforms.ToPILImage()(image)
        image_weak = torch.from_numpy(image_untrans_np.astype(np.float32))
        image_weak = transforms.ToPILImage()(image_weak)
        image_strong = torch.from_numpy(image_untrans_np.astype(np.float32))
        image_strong = transforms.ToPILImage()(image_strong)
        label = torch.from_numpy(label.astype(np.uint8))
        label = transforms.ToPILImage()(label)
        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "gt": label,
        }
        if self.split == 'train':
            transform_list = self.transform_list_train
        else:
            transform_list = self.transform_list_test

        sample_new = transform_list(sample)
        sample_new['image'] = torch.from_numpy(sample_new['image'].astype(np.float32)).unsqueeze(0)
        sample_new['image_weak'] = torch.from_numpy(sample_new['image_weak'].astype(np.float32)).unsqueeze(0)
        sample_new['image_strong'] = torch.from_numpy(sample_new['image_strong'].astype(np.float32)).unsqueeze(0)
        sample_new['label_aug'] = torch.from_numpy(sample_new['gt'].astype(np.uint8))

        return sample_new

    def resize(self, image):
        if len(image.shape) == 3:
            x, y, z = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        elif len(image.shape) == 2:
            x, y = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

# 新增 CGS使用
class OVRWeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size, num_classes):
        self.output_size = output_size
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)

        image_weak = self.resize(image)
        if random.random() > 0.5:
            image_weak, label = random_rot_flip(image_weak, label)
        elif random.random() > 0.5:
            image_weak, label = random_rotate(image_weak, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(
            image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        label_np = label.numpy()
        sample["ovr_label"] = []
        for i in range(self.num_classes - 1):
            # set label to 1 if it is the current class
            pix_2 = list(range(1, self.num_classes))
            if i+1 in pix_2:
                pix_2.remove(i+1)
            cur_label = np.zeros_like(label_np)
            cur_label[label_np == i+1] = 1
            for pix in pix_2:
                cur_label[label_np == pix] = 2
            sample["ovr_label"].append(
                torch.from_numpy(cur_label.astype(np.uint8)))

        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

# 新增 DECSeg 使用
class DECAugment(object):
    """returns ori, downsample and perturbation augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size
        self.totensor = transforms.ToTensor()

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image_ori, label_ori = self.resize(image, label)
        # weak augmentation is rotation / flip
        # image_ori, label_ori = random_rot_flip(image, label)
        image_down, label_down = self.down(image_ori, label_ori)
        # strong augmentation is color jitter
        image_per = color_jitter(image_ori).type("torch.FloatTensor")
        image_down_per = color_jitter(image_down).type("torch.FloatTensor")
        # fix dimensions
        image_ori = self.totensor(image_ori)
        image_down = self.totensor(image_down)
        label_ori = torch.from_numpy(label_ori.astype(np.uint8))
        label_down = torch.from_numpy(label_down.astype(np.uint8))

        sample_new = {
            "image": image_ori,
            "image_per": image_per,
            "image_down": image_down,
            "image_down_per": image_down_per,
            "label": label_ori,
            "label_down": label_down,
        }
        return sample_new

    def resize(self, image, label=None):
        # x, y, z = image.shape
        x, y= image.shape
        # image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        if label is not None:
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        return image, label

    def down(self, image, label=None):
        # image = zoom(image, (1 / 2, 1 / 2, 1), order=0)
        # if label is not None:
        #     label = zoom(label, (1 / 2, 1 / 2), order=0)
        outsize = round(self.output_size[0] * 0.5 / 32) * 32
        # image = zoom(image, (outsize / self.output_size[0], outsize / self.output_size[1], 1), order=0)
        image = zoom(image, (outsize / self.output_size[0], outsize / self.output_size[1]), order=0)
        if label is not None:
            label = zoom(label, (outsize / self.output_size[0], outsize / self.output_size[1]), order=0)
        return image, label

class WeakStrongAugment_3Dircadb(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size, args=None, split='train'):
        self.output_size = output_size
        self.split = split

        self.transform_list_train = {
            'c_random_flip': {'lr': True, 'ud': True},
            'c_random_rotate': {'range': [0, 359]},
            'c_random_image_enhance': {'methods': ['contrast', 'sharpness', 'brightness']},
            'c_random_gaussian_blur': {'apply': True},
            'c_tonumpy': None,
            'c_normalize': {'mean': None, 'std': None}, }
        self.transform_list_test = {
            'c_tonumpy': None,
            'c_normalize': {'mean': None, 'std': None}, }

        if args:
            if args.no_color:
                self.transform_list_train.pop('c_random_image_enhance', None)
            if args.no_blur:
                self.transform_list_train.pop('c_random_gaussian_blur', None)
            if args.rot != 359:
                self.transform_list_train['c_random_rotate']['range'] = [-args.rot, args.rot]
            # if

        self.transform_list_train = self.get_transform(self.transform_list_train)
        self.transform_list_test = self.get_transform(self.transform_list_test)

    @staticmethod
    def get_transform(transform_list):

        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __call__(self, sample):
        image_np = sample["image"].squeeze()  # (H, W) 去掉可能的 channel=1
        label_np = sample["label"].squeeze()  # (H, W)

        # resize 仍然用 numpy
        image_np = self.resize(image_np)
        label_np = self.resize(label_np)

        # 现在都是 2-D float32 / uint8
        image_t = torch.from_numpy(image_np.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        label_t = torch.from_numpy(label_np.astype(np.uint8))

        # 转 PIL
        image = F.to_pil_image(image_t)
        image_weak = F.to_pil_image(image_t)
        image_strong = F.to_pil_image(image_t)
        label = F.to_pil_image(label_t.unsqueeze(0))

        sample = {"image": image,
                  "image_weak": image_weak,
                  "image_strong": image_strong,
                  "gt": label}

        transform_list = self.transform_list_train if self.split == 'train' else self.transform_list_test
        sample_new = transform_list(sample)

        # 最后再转成 tensor 并补上 batch 维
        sample_new['image'] = torch.from_numpy(sample_new['image'].astype(np.float32)).unsqueeze(0)
        sample_new['image_weak'] = torch.from_numpy(sample_new['image_weak'].astype(np.float32)).unsqueeze(0)
        sample_new['image_strong'] = torch.from_numpy(sample_new['image_strong'].astype(np.float32)).unsqueeze(0)
        sample_new['label_aug'] = torch.from_numpy(sample_new['gt'].astype(np.uint8))
        sample_new['label'] = sample_new['label_aug']

        return sample_new

    def resize(self, image):
        if len(image.shape) == 3:
            x, y, z = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        elif len(image.shape) == 2:
            x, y = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
