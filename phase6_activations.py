import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# 1. Chargement et préparation
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

# On inclut les cas limites (linear) et adversariaux (softmax) dans la liste
activations = ['linear', 'sigmoid', 'tanh', 'relu', 'softmax']

results = []
histories = {}

print("Début des entraînements. Patientez quelques instants...\n")

for activation in activations:
    print(f"Entraînement avec l'activation : {activation}...")
    
    # 1. Fixer la seed pour une comparaison équitable
    tf.keras.utils.set_random_seed(42)
    
    # 2. Construction du modèle
    model = Sequential([
        Input(shape=(784,)),
        Dense(128, activation=activation),
        Dense(64, activation=activation),
        Dense(10, activation='softmax') # La sortie reste toujours softmax
    ])
    
    # 3. Compilation
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Entraînement
    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=0 # On masque les barres de progression pour garder un terminal propre
    )
    train_time_s = time.time() - start
    
    # 5. Évaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # 6. Trouver l'epoch de convergence (val_loss < 0.1)
    val_losses = history.history['val_loss']
    histories[activation] = val_losses
    
    convergence_epoch = "N/A"
    for epoch, v_loss in enumerate(val_losses):
        if v_loss < 0.1:
            convergence_epoch = epoch + 1 # +1 car les epochs commencent à 1
            break
            
    # 7. Stocker les résultats
    results.append({
        'activation': activation,
        'val_loss_final': val_losses[-1],
        'test_accuracy': test_acc,
        'convergence_epoch_sub01': convergence_epoch,
        'train_time_s': train_time_s
    })

# --- Affichage du Tableau Comparatif ---
print("\n" + "="*85)
print(f"{'Activation':12s} | {'Val loss epoch 10':18s} | {'Test accuracy':14s} | {'Epoch < 0.1 loss':16s} | {'Temps (s)':10s}")
print("-" * 85)
for r in results:
    print(f"{r['activation']:12s} | {r['val_loss_final']:18.4f} | {r['test_accuracy']:14.4f} | {str(r['convergence_epoch_sub01']):16s} | {r['train_time_s']:10.1f}")
print("="*85 + "\n")

# --- Tracé des Courbes Superposées ---
plt.figure(figsize=(10, 6))
# Couleurs spécifiques pour bien distinguer le bon grain de l'ivraie
colors = {'linear': 'gray', 'sigmoid': 'blue', 'tanh': 'orange', 'relu': 'green', 'softmax': 'red'}

for activation, val_losses in histories.items():
    plt.plot(range(1, 11), val_losses, label=activation, color=colors[activation], linewidth=2, marker='o', markersize=4)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Convergence selon la fonction d'activation (MNIST)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# On limite l'axe Y pour ne pas écraser les bonnes courbes à cause du softmax/linear
plt.ylim(0, 0.5) 
plt.savefig("graph/phase6_activations_curve.png", dpi=100, bbox_inches='tight')
plt.close()

print("Courbe sauvegardée : graph/phase6_activations_curve.png")