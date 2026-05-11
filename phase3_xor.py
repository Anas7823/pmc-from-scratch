import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Données d'entraînement XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

# Décommentez la ligne ci-dessous pour ajouter 5% de bruit.
# Un petit bruit peut perturber la convergence car la marge entre les classes est faible.
# X_xor += np.random.randn(*X_xor.shape) * 0.05

# 2. Fonctions utilitaires
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss_bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 3. Initialisation de l'architecture 2-2-1
# CAS LIMITE : Pour tester l'architecture 2-1-1, changez hidden_dim à 1.
# Explication : Avec 1 seul neurone caché, le réseau ne peut créer qu'une seule 
# "droite" d'activation. Il ne peut pas "plier" l'espace pour isoler les diagonales. 
# La loss stagnera et l'accuracy plafonnera autour de 50-75%.
input_dim = 2
hidden_dim = 2 
output_dim = 1

np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * 0.5
b1 = np.zeros(hidden_dim)

W2 = np.random.randn(hidden_dim, output_dim) * 0.5
b2 = np.zeros(output_dim)

learning_rate = 0.5
n_epochs = 10000
losses = []

# 4. Boucle d'entraînement
for epoch in range(n_epochs):
    # --- A. Forward pass ---
    # Couche 1 (cachée)
    z1 = np.dot(X_xor, W1) + b1
    a1 = sigmoid(z1)
    
    # Couche 2 (sortie)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    y_pred = a2.flatten()
    
    # --- B. Loss ---
    loss = compute_loss_bce(y_xor, y_pred)
    losses.append(loss)
    
    # --- C. Backpropagation ---
    # Erreur couche 2
    error2 = y_pred - y_xor
    dW2 = np.dot(a1.T, error2.reshape(-1, 1)) / len(y_xor)
    db2 = np.mean(error2)
    
    # Erreur couche 1 (Chain rule)
    error1 = np.dot(error2.reshape(-1, 1), W2.T) * a1 * (1 - a1)
    dW1 = np.dot(X_xor.T, error1) / len(y_xor)
    db1 = np.mean(error1, axis=0)
    
    # --- D. Mise à jour ---
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    # Affichage
    if epoch % 2000 == 0:
        acc = np.mean((y_pred > 0.5) == y_xor)
        print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

print(f"\nEntraînement terminé. Loss finale : {loss:.4f}")

# 5. Frontière de décision
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# On passe la grille dans le réseau entraîné
z1g = sigmoid(np.dot(grid, W1) + b1)
z2g = sigmoid(np.dot(z1g, W2) + b2)
Z = z2g.reshape(xx.shape)

plt.figure(figsize=(8, 6))
# On dessine les contours de probabilité (rouge = proche de 1, bleu = proche de 0)
plt.contourf(xx, yy, Z, levels=50, cmap="RdYlBu_r", alpha=0.8)
plt.colorbar(label="Probabilité prédite")

# On place les vrais points par-dessus
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, cmap="RdYlBu_r", edgecolors='k', s=150, linewidths=2)

plt.title("Frontière de décision (XOR) - Réseau 2-2-1")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.savefig("phase3_decision_boundary.png", dpi=100, bbox_inches='tight')
plt.close()

print("Courbe sauvegardée : phase3_decision_boundary.png")