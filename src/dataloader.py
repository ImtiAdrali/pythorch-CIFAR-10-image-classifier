import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import load_cifar10


def get_dataloader(batch_size=64, num_workers=0, seed=42):
    train_dataset, test_dataset = load_cifar10()

    train_size = 45000
    val_size = 5000

    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset = random_split(
        train_dataset,
        (train_size, val_size),
        generator=generator
    )

    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False
    )

    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader