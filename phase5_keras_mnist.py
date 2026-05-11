import tensorflow as tf
import keras
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

print(f"TensorFlow version: {tf.__version__}")

# 1. Chargement et préparation des données (MNIST)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Préprocessing : flatten 28x28 -> 784, normaliser entre 0 et 1
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

print(f"Train : {X_train.shape} | Test : {X_test.shape}")
print(f"Classes uniques : {np.unique(y_train)}")

# 2. Construction du modèle Keras
# Architecture : Entrée(784) -> Dense(128, relu) -> Dense(64, relu) -> Sortie(10, softmax)
model = keras.Sequential([
    # On définit explicitement l'entrée ici
    keras.layers.Input(shape=(784,)), 
    
    # Les couches Dense n'ont plus besoin de connaître l'entrée
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 3. Compilation
# - Adam : un optimiseur bien plus robuste que notre descente de gradient basique (SGD)
# - sparse_categorical_crossentropy : gère directement nos étiquettes (0 à 9) sans avoir besoin de faire du one-hot encoding
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Affichage de l'architecture et du nombre de paramètres (les fameux W et b !)
model.summary()

# 4. Entraînement
# SCÉNARIOS À TESTER :
# - Normal : epochs=5, batch_size=64
# - Limite : epochs=0 (L'historique sera vide)
# - Adversarial : batch_size=1 (L'entraînement sera atrocement long et bruité)
print("\n--- Début de l'entraînement ---")
start = time.time()

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=1,
    validation_split=0.1,
    verbose=1
)

elapsed = time.time() - start

# 5. Évaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTemps d'entraînement : {elapsed:.1f}s")
print(f"Test accuracy : {test_acc:.4f}")
print(f"Test loss : {test_loss:.4f}")

# 6. Tracé des courbes d'apprentissage
# Sécurité au cas où on teste avec epochs=0
if len(history.history.get('loss', [])) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Courbe de Loss
    axes[0].plot(history.history['loss'], label='train', marker='o')
    axes[0].plot(history.history['val_loss'], label='val', marker='o')
    axes[0].set_title("Loss (Entropie Croisée)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Courbe d'Accuracy
    axes[1].plot(history.history['accuracy'], label='train', marker='o')
    axes[1].plot(history.history['val_accuracy'], label='val', marker='o')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.savefig("graph/phase5_mnist_curves_adversarial.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("Courbes sauvegardées : phase5_mnist_curves_adversarial.png")
else:
    print("Aucune donnée d'entraînement à tracer (epochs=0).")