from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "runs" / "checkpoints"

def save_checkpoint(model, filename):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    filepath = CHECKPOINT_DIR / filename

    torch.save(model.state_dict(), filepath)

    print(f"Model saves to: {filepath}")

def load_checkpoint(model, filename, device):

    state_dict = torch.load(
        CHECKPOINT_DIR / filename,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)

    print(f"Model checkpoint loaded successfully!")

    return model

def save_best_checkpoint(model, val_loss, best_val_loss):

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        save_checkpoint(
            model,
            "best_model.pth"
        )

        print(
            f"New best model! "
            f"Validation loss: {best_val_loss:.4f}"
        )

    return best_val_loss

def save_training_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        best_val_loss,
        history,
        filename="last_checkpoint.pth"
):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    filename = CHECKPOINT_DIR / filename

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": best_val_loss,
        "history": history,
    }

    torch.save(checkpoint, filename)


def load_training_checkpoint(
        model,
        optimizer,
        scheduler,
        filepath,
        device
):
    checkpoint = torch.load(
        filepath,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    model.to(device)

    return {
        "epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["loss"],
        "history": checkpoint["history"],
    }