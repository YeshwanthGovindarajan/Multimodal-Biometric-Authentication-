import tensorflow as tf
import matplotlib.pyplot as plt

def load_model(model_path):
    """
    Load a pre-trained model from a given path.
    :param model_path: Path to the pre-trained model
    :return: Loaded Keras model
    """
    model = tf.keras.models.load_model(model_path)
    return model

def save_model(model, model_path):
    """
    Save the trained model to a given path.
    :param model: Trained Keras model
    :param model_path: Path to save the model
    """
    model.save(model_path)
    print(f"Model saved at {model_path}")

def plot_training_history(history):
    """
    Plot the training and validation accuracy and loss curves.
    :param history: History object from model.fit()
    """
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot the ROC curve using false positive rate (FPR) and true positive rate (TPR).
    :param fpr: False positive rate
    :param tpr: True positive rate
    :param roc_auc: Area under the curve (AUC)
    """
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
