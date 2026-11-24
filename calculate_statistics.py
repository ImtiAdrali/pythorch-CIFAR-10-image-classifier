import torch
from src.dataset import load_cifar10

train_dataset, _ = load_cifar10()

images = torch.stack([
    train_dataset[i][0]
    for i in range(len(train_dataset))
])

print(f"Dataset tensor shape: {images.shape}")

# Calculate mean and standard deviation per RGB channel
mean = images.mean(dim=(0, 2, 3))
std = images.std(dim=(0, 2, 3))

print(f"Channel means: {mean}")
print(f"Channel standard deviations: {std}")