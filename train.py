import torch
import torch.nn as nn

from models.CIFAR10Classifier import CIFAR10Classifier
from src.dataloader import get_dataloader
from src.engine import train_model

from src.checkpoint import load_training_checkpoint, CHECKPOINT_DIR

from src.visualization import plot_training_history

def main():

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    batch_size = 64
    learning_rate = 0.001
    epochs = 10

    train_loader, val_loader, test_loader = get_dataloader(batch_size=batch_size)
    model = CIFAR10Classifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=2,
        min_lr=1e-6,
    )

    checkpoint_path = CHECKPOINT_DIR / "last_checkpoint.pth"

    start_epoch = 0
    best_val_loss = float("inf")
    history = None

    if checkpoint_path.exists():

        checkpoint = load_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            filepath=checkpoint_path,
            device=device
        )

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        history = checkpoint["history"]

        print(f"Resuming from epoch {start_epoch + 1}")

    history = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=epochs,
        device=device,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        history=history
    )

    # TODO: Uncomment the following line to visualize the training history
    # plot_training_history(history=history)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()