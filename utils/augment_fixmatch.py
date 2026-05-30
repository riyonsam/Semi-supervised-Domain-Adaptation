"""
utils/augment.py  -  FixMatch augmentation pipeline for VisDA (224x224 images)

Implements:
  - RandAugment (Cubuk et al., 2020) -- used as the strong augmentation
  - cutout -- applied after RandAugment on strongly-augmented unlabeled images
  - FixMatchTransform -- single callable that returns one or two views of an image:
      * labeled / source:  returns a single weakly-augmented tensor
      * unlabeled target:  returns (weak_tensor, strong_tensor)
  - get_visda_transforms() -- builds the three transform objects needed for training

Augmentation design follows the FixMatch paper (Sohn et al., 2020):
  - Weak:    RandomCrop(224, padding=28) + RandomHorizontalFlip
  - Strong:  Weak + RandAugment(N=2, M=10) + Cutout(0.5 x image_size)
  - Test:    Resize(224x224) only  (deterministic)

All image operations work on PIL Images and are compatible with modern Pillow (>=10.0).
"""

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
from torchvision import transforms


# ImageNet normalisation constants  (VisDA uses ImageNet-pretrained ResNet-50)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# PIL-level augmentation operations  (used by RandAugment)
# Magnitude-bearing ops follow the ranges from the official FixMatch repo,
# adapted for Pillow >= 10.

def identity(img, _):
    return img

def translateX(img, mag):
    pixels = mag * img.size[0]
    return img.transform(img.size, Image.Transform.AFFINE,
                         (1, 0, pixels * random.choice([-1, 1]), 0, 1, 0))

def translateY(img, mag):
    pixels = mag * img.size[1]
    return img.transform(img.size, Image.Transform.AFFINE,
                         (1, 0, 0, 0, 1, pixels * random.choice([-1, 1])))

def shearX(img, mag):
    return img.transform(img.size, Image.Transform.AFFINE,
                         (1, mag * random.choice([-1, 1]), 0, 0, 1, 0))

def shearY(img, mag):
    return img.transform(img.size, Image.Transform.AFFINE,
                         (1, 0, 0, mag * random.choice([-1, 1]), 1, 0))

def rotate(img, mag):
    return img.rotate(mag * random.choice([-1, 1]))

def brightness(img, mag):
    return ImageEnhance.Brightness(img).enhance(mag)

def sharpness(img, mag):
    return ImageEnhance.Sharpness(img).enhance(mag)

def color(img, mag):
    return ImageEnhance.Color(img).enhance(mag)

def contrast(img, mag):
    return ImageEnhance.Contrast(img).enhance(mag)

def autocontrast(img, _):
    # Magnitude-free: equalise channel ranges
    return ImageOps.autocontrast(img)

def equalize(img, _):
    return ImageOps.equalize(img)

def posterize(img, mag):
    bits = max(1, min(8, int(mag)))
    return ImageOps.posterize(img, bits)

def solarize(img, mag):
    return ImageOps.solarize(img, int(mag))


# RandAugment operation table.
# Each entry: (function, range_min, range_max)
# None ranges mean the op ignores magnitude (e.g. equalize).
# Ranges from the official FixMatch / RandAugment implementation.
_RANDAUG_OPS = [
    (identity,      None,  None),
    (autocontrast,  None,  None),
    (equalize,      None,  None),
    (translateX,    0.0,   0.33),
    (translateY,    0.0,   0.33),
    (shearX,        0.0,   0.3),
    (shearY,        0.0,   0.3),
    (rotate,        0.0,   30.0),
    (brightness,    0.05,  0.95),
    (sharpness,     0.05,  0.95),
    (color,         0.05,  0.95),
    (contrast,      0.05,  0.95),
    (posterize,     4.0,   8.0),
    (solarize,      0.0,   256.0),
]


class RandAugment:
    """
    RandAugment (Cubuk et al., 2020).

    Randomly samples n operations from the augmentation list and applies them
    sequentially.  Magnitude m lives on a [0, 30] scale; FixMatch paper uses
    N=2, M=10.
    """

    def __init__(self, n=2, m=10):
        assert 0 <= m <= 30
        self.n = n
        self.m = m

    def __call__(self, img):
        ops = random.choices(_RANDAUG_OPS, k=self.n)
        for fn, lo, hi in ops:
            if lo is None:
                img = fn(img, None)
            else:
                mag = lo + (hi - lo) * self.m / 30.0
                img = fn(img, mag)
        return img


def cutout(img, mag=0.5):
    """
    Erase a random square patch (DeVries and Taylor, 2017).
    mag: side length of patch as fraction of image width (paper uses 0.5).
    """
    size = int(mag * img.size[0])
    if size <= 0:
        return img
    w, h = img.size
    cx = int(np.random.uniform(0, w))
    cy = int(np.random.uniform(0, h))
    x0 = max(0, cx - size // 2)
    y0 = max(0, cy - size // 2)
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size)
    img = img.copy()
    ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
    return img


class FixMatchTransform:
    """
    Applies one or two augmentation pipelines to a PIL image.

    - Labeled images  (strong=None): returns a single weakly-augmented tensor.
    - Unlabeled images (strong set): returns (weak_tensor, strong_tensor).
      weak view  -> pseudo-label generation (inside torch.no_grad)
      strong view -> cross-entropy loss
    """

    def __init__(self, weak, strong=None):
        self.weak   = weak
        self.strong = strong

    def __call__(self, img):
        if self.strong is None:
            return self.weak(img)
        return self.weak(img), self.strong(img)


_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_visda_transforms(img_size=224, cutout_mag=0.5):
    """
    Build the three FixMatch transform objects for VisDA.

    Weak augmentation  (paper Sec. 2):
        padding = 12.5% of img_size  (28 px for 224x224)
        RandomCrop(img_size, padding) + RandomHorizontalFlip

    Strong augmentation:
        same crop/flip + RandAugment(N=2, M=10) + Cutout

    Returns
    -------
    labeled_transform   -- FixMatchTransform, single weak view (tensor)
    unlabeled_transform -- FixMatchTransform, (weak_tensor, strong_tensor)
    test_transform      -- Compose, deterministic resize + normalize
    """
    padding = int(0.125 * img_size)   # 28 px for 224x224

    _weak_aug = transforms.Compose([
        transforms.RandomCrop(img_size, padding=padding, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ])

    _strong_aug = transforms.Compose([
        transforms.RandomCrop(img_size, padding=padding, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        RandAugment(n=2, m=10),
        transforms.Lambda(lambda img: cutout(img, cutout_mag)),
    ])

    labeled_transform = FixMatchTransform(
        weak=transforms.Compose([_weak_aug, _NORMALIZE]),
    )

    unlabeled_transform = FixMatchTransform(
        weak=transforms.Compose([_weak_aug, _NORMALIZE]),
        strong=transforms.Compose([_strong_aug, _NORMALIZE]),
    )

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        _NORMALIZE,
    ])

    return labeled_transform, unlabeled_transform, test_transform
