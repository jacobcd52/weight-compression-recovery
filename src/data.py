"""CIFAR-10 data loaders with standard augmentation."""
import os
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# Per-channel CIFAR-10 statistics (computed over the train split).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DEFAULT_ROOT = "/workspace/.cache/torch/datasets/cifar10"


def _train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_loaders(batch_size=128, num_workers=4, data_root=DEFAULT_ROOT,
                smoke=False, smoke_size=1024, augment=True):
    """Returns (train_loader, test_loader, num_classes).

    Downloads CIFAR-10 to `data_root` if not present. In smoke mode, both splits
    are subset to `smoke_size` images for a fast end-to-end pipeline check.
    """
    os.makedirs(data_root, exist_ok=True)

    train_tf = _train_transform() if augment else _test_transform()
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_test_transform())

    if smoke:
        train_set = Subset(train_set, list(range(min(smoke_size, len(train_set)))))
        test_set = Subset(test_set, list(range(min(smoke_size, len(test_set)))))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=num_workers > 0)
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0)

    return train_loader, test_loader, 10


if __name__ == "__main__":
    tr, te, nc = get_loaders(smoke=True)
    xb, yb = next(iter(tr))
    print("train batch:", tuple(xb.shape), "labels:", tuple(yb.shape), "classes:", nc)
    print("train images:", len(tr.dataset), "test images:", len(te.dataset))
