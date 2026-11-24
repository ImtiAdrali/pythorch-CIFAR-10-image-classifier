import torch
from models.CIFAR10Classifier import CIFAR10Classifier
from src.dataloader import get_dataloader

# Load one batch actual CIFAR-10 images
train_loader, test_loader = get_dataloader(batch_size=64)
images, labels = next(iter(train_loader))


# Create the mode
model = CIFAR10Classifier()

# Forward pass
output = model(images)

# Display information
print(f"Input shape: {images.shape}")
print(f"Output shape: {output.shape}")

print(f"\nModel architecture:")
print(model)

print(f"\nModel Parameters:")

for name, parameter in model.named_parameters():
    print(name, parameter, parameter.shape)

# Count trainable parameters
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"\nTotal number of trainable parameters: {total_parameters}")

# Verify
assert output.shape == (64, 10)
assert total_parameters == 402250

print("\nModel verification successful!")