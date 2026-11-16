"""Paired image/mask resizing, normalization, and safe geometric augmentation."""

from __future__ import annotations

import random

import torch
from PIL import Image, ImageOps


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an 8-bit PIL image to a channel-first float tensor."""
    image = image.copy()
    channels = len(image.getbands())
    raw = bytearray(image.tobytes())
    tensor = torch.frombuffer(raw, dtype=torch.uint8)
    if channels == 1:
        tensor = tensor.reshape(image.height, image.width).unsqueeze(0)
    else:
        tensor = tensor.reshape(image.height, image.width, channels).permute(2, 0, 1)
    return tensor.to(dtype=torch.float32).div_(255.0)


class ImageTransform:
    """Resize RGB images and apply ImageNet channel normalization."""

    def __init__(self, image_size: int | tuple[int, int], *, normalize: bool = True):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2 or min(image_size) < 1:
            raise ValueError("image_size must contain two positive dimensions")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.normalize = normalize

    def __call__(self, image: Image.Image) -> torch.Tensor:
        width, height = self.image_size[1], self.image_size[0]
        image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        image_tensor = _to_tensor(image)
        if self.normalize:
            mean = image_tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
            std = image_tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        return image_tensor


class MaskTransform:
    """Resize ground-truth masks with nearest-neighbor sampling and binarize."""

    def __init__(self, image_size: int | tuple[int, int]):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2 or min(image_size) < 1:
            raise ValueError("image_size must contain two positive dimensions")
        self.image_size = (int(image_size[0]), int(image_size[1]))

    def __call__(self, mask: Image.Image) -> torch.Tensor:
        width, height = self.image_size[1], self.image_size[0]
        mask = mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
        return (_to_tensor(mask) > 0).to(dtype=torch.float32)


class PairedTransform:
    """Resize image and mask together; only training instances may augment."""

    def __init__(
        self,
        image_size: int | tuple[int, int],
        *,
        training: bool = False,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        normalize: bool = True,
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2 or min(image_size) < 1:
            raise ValueError("image_size must contain two positive dimensions")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.training = training
        self.horizontal_flip = horizontal_flip and training
        self.vertical_flip = vertical_flip and training
        self.normalize = normalize
        self.image_transform = ImageTransform(self.image_size, normalize=normalize)
        self.mask_transform = MaskTransform(self.image_size)

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.horizontal_flip and random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if self.vertical_flip and random.random() < 0.5:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        return self.image_transform(image), self.mask_transform(mask)


def training_transform(
    image_size: int | tuple[int, int],
    *,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
) -> PairedTransform:
    """Create the train-only transform with opt-in flips (disabled by default)."""
    return PairedTransform(
        image_size,
        training=True,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
    )


def evaluation_transform(image_size: int | tuple[int, int]) -> PairedTransform:
    """Create a deterministic evaluation transform with no augmentation."""
    return PairedTransform(image_size, training=False)
