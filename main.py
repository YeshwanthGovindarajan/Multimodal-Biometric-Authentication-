import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

def load_lfw_dataset():
    print("Loading LFW (face) dataset...")
    (train_face, train_labels), (test_face, test_labels) = tf.keras.datasets.lfw.load_data()  # Placeholder for LFW
    return train_face, train_labels, test_face, test_labels

def load_voxceleb_dataset():
    print("Loading VoxCeleb (voice) dataset...")
    # Assuming VoxCeleb data is stored in a directory in spectrogram format
    train_voice = image_dataset_from_directory('path_to_voxceleb/train', image_size=(128, 128), color_mode='grayscale')
    test_voice = image_dataset_from_directory('path_to_voxceleb/test', image_size=(128, 128), color_mode='grayscale')
    return train_voice, test_voice

def load_mcyt100_dataset():
    print("Loading MCYT-100 (signature) dataset...")
    # Assuming signature data is stored in a directory of images
    train_sig = image_dataset_from_directory('path_to_mcyt100/train', image_size=(224, 224), color_mode='grayscale')
    test_sig = image_dataset_from_directory('path_to_mcyt100/test', image_size=(224, 224), color_mode='grayscale')
    return train_sig, test_sig

def preprocess_data(train_face, train_voice, train_sig):
    print("Preprocessing datasets...")

    train_face = train_face / 255.0
    train_voice = train_voice / 255.0
    train_sig = train_sig / 255.0

    return train_face, train_voice, train_sig

def shared_cnn_layers(input_layer):
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x

def modality_specific_layers(input_layer, modality):
    if modality == 'face' or modality == 'signature':
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        return x
    elif modality == 'voice':
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Reshape((128, -1))(x)  # Reshape for LSTM
        x = layers.LSTM(64, return_sequences=False)(x)
        return x

def build_multimodal_model():
    print("Building multimodal model...")


    face_input = layers.Input(shape=(224, 224, 3), name='face_input')
    face_shared = shared_cnn_layers(face_input)
    face_specific = modality_specific_layers(face_shared, 'face')

    sig_input = layers.Input(shape=(224, 224, 1), name='signature_input')
    sig_shared = shared_cnn_layers(sig_input)
    sig_specific = modality_specific_layers(sig_shared, 'signature')

    voice_input = layers.Input(shape=(128, 128, 1), name='voice_input')
    voice_specific = modality_specific_layers(voice_input, 'voice')

    concatenated = layers.Concatenate()([face_specific, sig_specific, voice_specific])

    fc1 = layers.Dense(128, activation='relu')(concatenated)
    fc2 = layers.Dense(64, activation='relu')(fc1)
    output = layers.Dense(1, activation='sigmoid')(fc2)

    model = models.Model(inputs=[face_input, sig_input, voice_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, train_face, train_sig, train_voice, labels):
    print("Training the model...")

    history = model.fit([train_face, train_sig, train_voice], labels, epochs=50, batch_size=32, validation_split=0.2)

    return model, history

def evaluate_model(model, test_face, test_sig, test_voice, test_labels):
    print("Evaluating the model...")

    loss, accuracy = model.evaluate([test_face, test_sig, test_voice], test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    predictions = model.predict([test_face, test_sig, test_voice])
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    return accuracy

def main():

    train_face, train_labels, test_face, test_labels = load_lfw_dataset()
    train_voice, test_voice = load_voxceleb_dataset()
    train_sig, test_sig = load_mcyt100_dataset()

    train_face, train_voice, train_sig = preprocess_data(train_face, train_voice, train_sig)

    labels = train_labels  # Use multimodal labels when available

   model = build_multimodal_model()

    model, history = train_model(model, train_face, train_sig, train_voice, labels)

    accuracy = evaluate_model(model, test_face, test_sig, test_voice, test_labels)

if __name__ == "__main__":
    main()
