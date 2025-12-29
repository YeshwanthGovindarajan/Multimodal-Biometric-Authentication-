import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


from tensorflow.keras.utils import image_dataset_from_directory
import librosa
import numpy as np

def load_lfw_dataset():
    print("Download LFW manually and load using your local path")
    raise NotImplementedError("No TF built-in LFW loader")

def load_voxceleb_dataset():
    print("Loading VoxCeleb audio dataset...")
    def load_audio_folder(folder):
        X, y = [], []
        for label, person in enumerate(os.listdir(folder)):
            person_path = os.path.join(folder, person)
            for file in os.listdir(person_path):
                wav, sr = librosa.load(os.path.join(person_path, file), sr=16000)
                mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40)
                X.append(np.mean(mfcc, axis=1))
                y.append(label)
        return np.array(X), np.array(y)
    X_train, y_train = load_audio_folder("path_to_voxceleb/train")
    X_test, y_test = load_audio_folder("path_to_voxceleb/test")
    return X_train, y_train, X_test, y_test

def load_mcyt100_dataset():
    print("Load MCYT-100 using your local directory...")
    train_sig = image_dataset_from_directory("path_to_mcyt100/train", image_size=(224,224), color_mode='grayscale')
    test_sig = image_dataset_from_directory("path_to_mcyt100/test", image_size=(224,224), color_mode='grayscale')
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
    if modality in ['face', 'signature']:
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        return x
    elif modality == 'voice':
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)
        # reshape for LSTM: (batch, time, features)
        x = layers.Reshape((x.shape[1], -1))(x)
        x = layers.LSTM(64, return_sequences=False)(x)
        return x
    else:
        raise ValueError(f"Unknown modality: {modality}")

def build_feature_extractor(embedding_dim=128):
    """
    Multimodal feature extractor.
    Outputs a single embedding vector per (face, signature, voice) triplet.
    """
    #224x224x3
    face_input = layers.Input(shape=(224, 224, 3), name='face_input')
    face_shared = shared_cnn_layers(face_input)
    face_specific = modality_specific_layers(face_shared, 'face')

    #224x224x1
    sig_input = layers.Input(shape=(224, 224, 1), name='signature_input')
    sig_shared = shared_cnn_layers(sig_input)
    sig_specific = modality_specific_layers(sig_shared, 'signature')

    # 128x128x1
    voice_input = layers.Input(shape=(128, 128, 1), name='voice_input')
    voice_shared = shared_cnn_layers(voice_input)
    voice_specific = modality_specific_layers(voice_shared, 'voice')

    concatenated = layers.Concatenate(name="concat_features")(
        [face_specific, sig_specific, voice_specific]
    )

    embedding = layers.Dense(embedding_dim, activation='relu', name='embedding')(concatenated)
    output = layers.Dense(1, activation='sigmoid', name='auth_output')(embedding)

    model = models.Model(
        inputs=[face_input, sig_input, voice_input],
        outputs=output,
        name="multimodal_auth_net"
    )
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def compute_far_frr_eer(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    FAR = fpr
    FRR = 1 - tpr
    idx_eer = np.nanargmin(np.absolute(FAR - FRR))
    eer = (FAR[idx_eer] + FRR[idx_eer]) / 2.0
    eer_threshold = thresholds[idx_eer]

    return FAR, FRR, eer, eer_threshold, fpr, tpr


def train_feature_extractor(model,
                            train_face, train_sig, train_voice, y_train,
                            val_split=0.2,
                            epochs=20,
                            batch_size=32):
    history = model.fit(
        [train_face, train_sig, train_voice],
        y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history


def build_embedding_model(trained_model):
    embedding_layer = trained_model.get_layer('embedding')
    embedding_model = models.Model(
        inputs=trained_model.inputs,
        outputs=embedding_layer.output,
        name="embedding_extractor"
    )
    return embedding_model

def extract_embeddings(embedding_model, face, sig, voice, batch_size=32):
    embeddings = embedding_model.predict(
        [face, sig, voice],
        batch_size=batch_size,
        verbose=1
    )
    return embeddings


def train_pca_gbm(train_embeddings, y_train, n_components=32):
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_reduced = pca.fit_transform(train_embeddings)
    gbm = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    gbm.fit(X_train_reduced, y_train)

    return pca, gbm

def evaluate_pca_gbm(pca, gbm, test_embeddings, y_test):
    X_test_reduced = pca.transform(test_embeddings)
    y_scores = gbm.predict_proba(X_test_reduced)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    FAR, FRR, eer, eer_threshold, fpr_full, tpr_full = compute_far_frr_eer(y_test, y_scores)

    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer * 100:.4f}% at threshold {eer_threshold:.4f}")
    idx_eer = np.nanargmin(np.absolute(FAR - FRR))
    print(f"FAR at EER: {FAR[idx_eer] * 100:.4f}%")
    print(f"FRR at EER: {FRR[idx_eer] * 100:.4f}%")

    plt.figure()
    plt.plot(fpr_full, tpr_full, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("ROC Curve - Multimodal PCA + GBM")
    plt.legend()
    plt.grid(True)
    plt.show()

    return roc_auc, eer, FAR, FRR, y_scores


def main():

    train_face, train_labels, test_face, test_labels = load_lfw_dataset()
    train_voice, test_voice = load_voxceleb_dataset()
    train_sig, test_sig = load_mcyt100_dataset()
    train_face, train_voice, train_sig = preprocess_data(train_face, train_voice, train_sig)
    labels = train_labels

    model = build_feature_extractor(embedding_dim=128)
    print(model.summary())
    train_feature_extractor(
        model,
        train_face, train_sig, train_voice,
        y_train,
        epochs=20,
        batch_size=32
    )

    embedding_model = build_embedding_model(model)
    train_embeddings = extract_embeddings(embedding_model, train_face, train_sig, train_voice)
    test_embeddings = extract_embeddings(embedding_model, test_face, test_sig, test_voice)

    pca, gbm = train_pca_gbm(train_embeddings, y_train, n_components=32)
    roc_auc, eer, FAR, FRR, y_scores = evaluate_pca_gbm(pca, gbm, test_embeddings, y_test)

if __name__ == "__main__":
    main()

