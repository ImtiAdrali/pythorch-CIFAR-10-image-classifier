from pathlib import Path
from torchvision.datasets import CIFAR10
from src.transform import get_transform

# Locate the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset storage directory
DATA_DIR = PROJECT_ROOT / "data"

def load_cifar10(download=True):
    """
    Load CIFAR10 training and test datasets
    """
    transform = get_transform()

    train_dataset = CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=download,
        transform=transform
    )

    test_dataset = CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=download,
        transform=transform
    )

    return train_dataset, test_dataset