import torch
from src.checkpoint import save_best_checkpoint, save_training_checkpoint

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:

        # Move data to the selected device
        images = images.to(device)
        labels = labels.to(device)

        # Clean gradients from the previous iteration
        optimizer.zero_grad()

        # Forward pass
        output = model(images)

        # Calculate loss
        loss = criterion(output, labels)

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Track loss
        running_loss += loss.item() * images.size(0)

        # Calculate predictions
        predictions = output.argmax(dim=1)

        # Track correct predictions
        correct_predictions += (predictions == labels).sum().item()

        # Update total samples
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = criterion(output, labels)

            running_loss += loss.item() * images.size(0)

            predictions = output.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()

            total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy

def train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs,
        start_epoch=0,
        best_val_loss=float("inf"),
        history=None
):

    if history is None:
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    for epoch in range(start_epoch, epochs):

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        best_val_loss = save_best_checkpoint(
            model=model,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
        )
        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            history=history
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1} / {epochs}] ] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy * 100:.2f} "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy * 100:.2f} | "
            f"LR: {current_lr:.6f}"
        )


    return history
