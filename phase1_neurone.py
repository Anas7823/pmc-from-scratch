import numpy as np

print(f"NumPy version : {np.__version__}")

# Données d'entraînement
X = np.array([
    [0.2, 0.1],
    [0.8, 0.9],
    [0.3, 0.7],
    [0.9, 0.2],
])
y = np.array([0, 1, 1, 0])

# Implémentation de la fonction d'activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Implémentation de la passe avant (forward)
def forward(X, w, b):
    # Étape 1 : somme pondérée
    z = np.dot(X, w) + b
    # Étape 2 : activation
    return sigmoid(z)

# Fonction pour calculer l'erreur (Binary Cross-Entropy)
def binary_cross_entropy(y_true, y_pred):
    # On ajoute un petit epsilon pour éviter de calculer log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ==========================================
# TESTS DES SCÉNARIOS
# ==========================================

print("\n--- Scénario normal ---")
# Poids fixés manuellement pour donner un résultat correct sur ce dataset
# (On pénalise la feature 1 et on récompense la feature 2)
w_normal = np.array([-1.5, 2.5])
b_normal = -0.5

preds_normal = forward(X, w_normal, b_normal)
loss_normal = binary_cross_entropy(y, preds_normal)
print(f"Prédictions : {preds_normal.round(3)}")
print(f"Loss BCE    : {loss_normal:.4f}")


print("\n--- Cas limite (X rempli de zéros) ---")
# Le biais seul (-0.5) pilote la sortie : sigmoid(-0.5) = 0.377
X_zeros = np.zeros((4, 2))
preds_limite = forward(X_zeros, w_normal, b_normal)
print(f"Prédictions : {preds_limite.round(3)}")


print("\n--- Scénario adversarial (poids et biais à 0) ---")
# Le pire point de départ possible
w_zero = np.zeros(2)
b_zero = 0.0

preds_adv = forward(X, w_zero, b_zero)
loss_adv = binary_cross_entropy(y, preds_adv)
print(f"Prédictions : {preds_adv}")
print(f"Loss BCE    : {loss_adv:.4f}") # Doit afficher ~0.693 (qui est -log(0.5))