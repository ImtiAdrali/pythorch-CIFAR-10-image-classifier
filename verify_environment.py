import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Selected device: {device}")

x = torch.randn(1000, 1000, device=device)
result = x @ x.T

print(f"Result shape: {result.shape}")
print(f"Tensor device: {result.device}")
print(f"Verification successful!")