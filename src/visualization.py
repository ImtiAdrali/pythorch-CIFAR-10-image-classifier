import matplotlib.pyplot as plt

def plot_training_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label ="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label ="Validation Loss"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.show()

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        [accuracy * 100 for accuracy in history["train_accuracy"]],
        label ="Training Accuracy"
    )

    plt.plot(
        epochs,
        [accuracy * 100 for accuracy in history["val_accuracy"]],
        label ="Validation Accuracy"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")

    plt.legend()
    plt.grid(True)

    plt.show()