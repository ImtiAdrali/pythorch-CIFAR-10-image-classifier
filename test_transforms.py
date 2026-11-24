import torch
from src.dataset import load_cifar10

train_dataset, test_dataset = load_cifar10()

image, label = train_dataset[0]

print("Image type:", type(image))
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Image minimum:", image.min().item())
print("Image maximum:", image.max().item())

print("Image mean:", image.mean().item())
print("Label:", label)

assert isinstance(image, torch.Tensor)
assert image.shape == (3, 32, 32)
assert image.dtype == torch.float32

print("Normalization verification successful!")
