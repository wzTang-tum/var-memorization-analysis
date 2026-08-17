"""
Data loading and augmentation.

Reuses VAR's own train transform (utils/data.py's build_dataset train_aug
branch) rather than reinventing it: Resize(288, LANCZOS) -> RandomCrop(256)
-> ToTensor -> normalize to [-1,1]. No horizontal flip, matching how the
released VAR-d16 checkpoint was trained.

Decode-once -> N views -> release: never hold more than one decoded PIL
image in memory at a time.
"""
import json
import os

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, transforms

from config import SEED  # noqa: F401 (re-exported for convenience)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_RESO = 256
MID_RESO = 1.125


def build_train_transform(hflip: bool = False) -> transforms.Compose:
    """Matches VAR's utils/data.py build_dataset() train_aug branch."""
    mid_reso_px = round(MID_RESO * FINAL_RESO)
    aug = [
        transforms.Resize(mid_reso_px, interpolation=InterpolationMode.LANCZOS),
        transforms.RandomCrop((FINAL_RESO, FINAL_RESO)),
        transforms.ToTensor(),
        lambda x: x.add(x).add_(-1),  # normalize_01_into_pm1
    ]
    if hflip:
        aug.insert(0, transforms.RandomHorizontalFlip())
    return transforms.Compose(aug)


TRAIN_TRANSFORM = build_train_transform(hflip=False)


def load_subset(path=None):
    path = path or os.path.join(ROOT, 'data', 'subset_1000.json')
    with open(path, encoding='utf-8') as f:
        manifest = json.load(f)
    return manifest


def make_views(pil_img: Image.Image, n_views: int = 10, transform=None) -> torch.Tensor:
    """decode-once -> n_views augmented crops -> [n_views, 3, 256, 256], range [-1, 1]."""
    transform = transform or TRAIN_TRANSFORM
    views = torch.stack([transform(pil_img) for _ in range(n_views)])
    return views


def load_image_views(local_path: str, n_views: int = 10, transform=None) -> torch.Tensor:
    """Decode once, generate views, release the PIL image."""
    with Image.open(os.path.join(ROOT, local_path)) as im:
        im = im.convert('RGB')
        views = make_views(im, n_views=n_views, transform=transform)
    return views
