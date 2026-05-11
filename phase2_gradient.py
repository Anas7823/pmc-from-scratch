import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Données d'entraînement
X = np.array([[0.2, 0.1], [0.8, 0.9], [0.3, 0.7], [0.9, 0.2]])
y = np.array([0, 1, 1, 0])

# 2. Fonctions utilitaires
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 3. Initialisation
np.random.seed(42)
w = np.random.randn(2) * 0.01  # Petits poids aléatoires pour briser la symétrie
b = 0.0

learning_rate = 0.1  # N'hésitez pas à tester 0 ou 10.0 pour les scénarios !
n_epochs = 50
losses = []

# 4. Boucle d'entraînement
for epoch in range(n_epochs):
    # --- A. Forward pass ---
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    
    # --- B. Calcul de la Loss ---
    loss = compute_loss(y, y_pred)
    losses.append(loss)
    
    # --- C. Backpropagation (Calcul des gradients) ---
    # La dérivée de (BCE + Sigmoid) se simplifie magnifiquement en (Prédiction - Cible)
    error = y_pred - y
    
    # Gradient par rapport aux poids : Moyenne pondérée par les entrées X
    dw = np.dot(X.T, error) / len(y)
    
    # Gradient par rapport au biais : Moyenne simple de l'erreur
    db = np.mean(error)
    
    # --- D. Mise à jour (Descente de gradient) ---
    # On soustrait le gradient pour descendre vers le minimum de la loss
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # Affichage de progression
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | w: {w.round(3)} | b: {b:.3f}")

# 5. Visualisation
plt.figure(figsize=(8, 4))
plt.plot(losses, color='blue', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss BCE")
plt.title(f"Convergence du neurone unique (LR={learning_rate})")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("phase2_loss_curve.png", dpi=100, bbox_inches='tight')
plt.close()

print(f"\nCourbe sauvegardée : phase2_loss_curve.png")
print(f"Loss finale : {losses[-1]:.4f}")