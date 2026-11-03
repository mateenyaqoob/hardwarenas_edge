from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def get_cifar10(batch=128, root="data", fake=False, split_search=True, n_fake=256):
    """Returns (loader_a, loader_b): (weights-split, arch-split) if split_search else (train, test).

    fake=True uses random tensors (smoke tests only; no download, no torchvision)."""
    if fake:
        mk = lambda n: TensorDataset(torch.randn(n, 3, 32, 32), torch.randint(0, 10, (n,)))
        train, test = mk(n_fake), mk(n_fake // 2)
    else:
        from torchvision import datasets, transforms as T
        norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        train = datasets.CIFAR10(root, True, download=True, transform=T.Compose(
            [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm]))
        test = datasets.CIFAR10(root, False, download=True, transform=T.Compose([T.ToTensor(), norm]))
    if split_search:
        n = len(train) // 2
        a, b = random_split(train, [n, len(train) - n], generator=torch.Generator().manual_seed(0))
        return DataLoader(a, batch, shuffle=True, drop_last=True), DataLoader(b, batch, shuffle=True, drop_last=True)
    return DataLoader(train, batch, shuffle=True, drop_last=True), DataLoader(test, 256)
