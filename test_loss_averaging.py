import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

logits = torch.tensor([
    [2.0, 1.0, 0.0],
    [1.0, 2.0, 0.0],
    [0.0, 2.0, 1.0],
    [0.0, 1.0, 2.0],
    [2.0, 0.0, 1.0]
])

labels = torch.tensor([0, 1, 1, 2, 0])

loss1 = criterion(logits[:2], labels[:2])
loss2 = criterion(logits[2:4], labels[2:4])
loss3 = criterion(logits[4:], labels[4:])

# Incorrect: average the three batches
wrong_epoch_loss = (loss1 + loss2 + loss3) / 3

# Correct: weight each batch average by it size
total_loss = (
    loss1.item() * 2 + loss2.item() * 2 + loss3.item() * 2
)
correct_epoch_loss = total_loss / 5

# Calculate the loss directly across all five images
direct_epoch_loss = criterion(logits, labels).item()

print("Batch 1 loss:", loss1.item())
print("Batch 2 loss:", loss2.item())
print("Batch 3 loss:", loss3.item())

print("\nIncorrect epoch loss:", wrong_epoch_loss.item())
print("Correct epoch loss:", correct_epoch_loss)
print("Direct epoch loss:", direct_epoch_loss)

assert abs(correct_epoch_loss - direct_epoch_loss) < 1e-6

print("\nLoss averaging verification successful!")