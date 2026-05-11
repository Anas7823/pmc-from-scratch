import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. GÉNÉRATION DES DONNÉES
# ==========================================
def generate_spiral(n_points=200, noise=0.1, seed=42):
    """Génère deux spirales entrelacées : classe 0 et classe 1."""
    np.random.seed(seed)
    n = n_points // 2
    theta0 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise
    theta1 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise + np.pi
    r = np.linspace(0.1, 1.0, n)
    X0 = np.c_[r * np.cos(theta0), r * np.sin(theta0)]
    X1 = np.c_[r * np.cos(theta1), r * np.sin(theta1)]
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y

# SCÉNARIOS À TESTER :
# - Normal : noise = 0.15
# - Adversarial : noise = 0.5 (Les classes se chevauchent, l'accuracy va chuter)
NOISE_LEVEL = 0.15 
X, y = generate_spiral(n_points=400, noise=NOISE_LEVEL)


# ==========================================
# 2. FONCTIONS D'ACTIVATION ET LOSS
# ==========================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ==========================================
# 3. INITIALISATION DU RÉSEAU (2-64-64-1)
# ==========================================
# CAS LIMITE : Pour observer l'underfitting, changez H1 et H2 à 2.
H1 = 64
H2 = 64

np.random.seed(42)
# Initialisation de He : std = sqrt(2 / n_entrées)
W1 = np.random.randn(2, H1) * np.sqrt(2.0 / 2)
b1 = np.zeros(H1)

W2 = np.random.randn(H1, H2) * np.sqrt(2.0 / H1)
b2 = np.zeros(H2)

W3 = np.random.randn(H2, 1) * np.sqrt(2.0 / H2)
b3 = np.zeros(1)


# ==========================================
# 4. BOUCLE D'ENTRAÎNEMENT
# ==========================================
lr = 0.5  # Taux d'apprentissage ajusté pour converger plus vite
n_epochs = 2000
losses = []
n = len(y)

print(f"--- Entraînement 2-{H1}-{H2}-1 (Bruit: {NOISE_LEVEL}) ---")

for epoch in range(n_epochs):
    # --- Forward Pass ---
    # Couche 1
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    
    # Couche 2
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    
    # Couche 3 (Sortie)
    z3 = np.dot(a2, W3) + b3
    y_pred = sigmoid(z3).flatten()
    
    # Calcul de la Loss
    loss = bce_loss(y, y_pred)
    losses.append(loss)
    
    # --- Backward Pass (Rétropropagation) ---
    # Gradient Sortie
    err3 = y_pred - y
    dW3 = np.dot(a2.T, err3.reshape(-1, 1)) / n
    db3 = np.mean(err3)
    
    # Gradient Couche 2 (Chain Rule avec dérivée de ReLU)
    err2 = np.dot(err3.reshape(-1, 1), W3.T) * relu_grad(z2)
    dW2 = np.dot(a1.T, err2) / n
    db2 = np.mean(err2, axis=0)
    
    # Gradient Couche 1
    err1 = np.dot(err2, W2.T) * relu_grad(z1)
    dW1 = np.dot(X.T, err1) / n
    db1 = np.mean(err1, axis=0)
    
    # --- Mise à jour des poids ---
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3
    
    if epoch % 500 == 0 or epoch == n_epochs - 1:
        acc = np.mean((y_pred > 0.5) == y)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

print(f"\nLoss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {np.mean((y_pred > 0.5) == y):.2%}")

# ==========================================
# 5. VISUALISATION
# ==========================================
h = 0.02
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]

# Prédiction sur la grille pour dessiner la frontière
a1g = relu(np.dot(grid, W1) + b1)
a2g = relu(np.dot(a1g, W2) + b2)
zg = sigmoid(np.dot(a2g, W3) + b3).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Frontière
axes[0].contourf(xx, yy, zg, alpha=0.4, cmap='RdBu', levels=50)
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', s=20, edgecolors='k')
axes[0].set_title(f"Frontière de décision (2-{H1}-{H2}-1)")

# Plot Loss
axes[1].plot(losses, color='purple', linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss BCE")
axes[1].set_title("Courbe de loss spirale")
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.savefig("graph/phase4_spirale.png", dpi=100, bbox_inches='tight')
plt.close()

print("\nCourbe sauvegardée : phase4_spirale.png")