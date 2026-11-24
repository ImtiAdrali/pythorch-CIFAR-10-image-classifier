from src.dataloader import get_dataloader

train_loader, val_loader, test_loader = get_dataloader(batch_size=64, num_workers=0)
images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("Image dtype:", images.dtype)
print("Label dtype:", labels.dtype)

print("Number of training batches:", len(train_loader))
print("Number of testing batches:", len(test_loader))
print("Training shuffle:", train_loader.sampler)
print("Training drop_last:", train_loader.drop_last)
print("Training workers:", train_loader.num_workers)
print("Training pin_memory:", train_loader.pin_memory)

print("\nVerifiy training and validation split:")
print("Training samples:", len(train_loader.dataset))
print("Validation samples:", len(val_loader.dataset))
print("Testing samples:", len(test_loader.dataset))

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches:", len(test_loader))