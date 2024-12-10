import matplotlib.pyplot as plt

def plot_results(histories, labels):
    plt.figure(figsize=(12, 8))
    for history, label in zip(histories, labels):
        plt.plot(history.history['val_accuracy'], label=f"{label} Validation Accuracy")
    plt.title("Validation Accuracy for Different Optimizers")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
