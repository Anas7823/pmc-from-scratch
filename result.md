# 🧠 Rapport d'Expérimentation : Phase 1 - Neurone Artificiel

> [!NOTE]
> **Objectif :** Implémentation "from scratch" des fondations d'un Perceptron Multi-Couche (PMC), incluant la passe avant (forward pass) et le calcul de l'erreur via la Binary Cross-Entropy (BCE).

---

## 🏗️ Architecture du Modèle
Le modèle testé dans cette phase est un neurone unique (ou couche dense de sortie) configuré pour une tâche de classification binaire.

```mermaid
graph LR
    X1((Input 1)) -- w1 --> Sum[Σ / Sum]
    X2((Input 2)) -- w2 --> Sum
    B((Biais)) -- 1 --> Sum
    Sum --> Act{σ / Sigmoid}
    Act --> Out((Prédiction))
    
    style Sum fill:#f9f,stroke:#333,stroke-width:2px
    style Act fill:#bbf,stroke:#333,stroke-width:2px
    style Out fill:#bfb,stroke:#333,stroke-width:2px
```

---

## 📊 Analyse des Scénarios d'Exécution

### 1. Scénario Normal
*Configuration : Poids optimisés manuellement ($w = [-1.5, 2.5]$, $b = -0.5$)*

| Échantillon | Prédiction | Label Attendu |
| :--- | :--- | :--- |
| n°1 | **0.366** | 0 |
| n°2 | **0.634** | 1 |
| n°3 | **0.690** | 1 |
| n°4 | **0.206** | 0 |

> [!TIP]
> **Performance :** La perte BCE est de **0.3781**, indiquant que le modèle commence déjà à séparer les classes avec ces poids.

---

### 2. Cas Limite (Inputs à Zéro)
*Configuration : Entrées $X = [0, 0]$, mêmes poids que ci-dessus.*

| Métrique | Valeur |
| :--- | :--- |
| Prédictions | `[0.378, 0.378, 0.378, 0.378]` |

**Observation :** En l'absence de signal d'entrée, la sortie est uniquement pilotée par le biais. $\sigma(-0.5) \approx 0.378$.

---

### 3. Scénario Adversarial (Initialisation à Zéro)
*Configuration : Poids et biais initialisés à $0$.*

| Métrique | Valeur |
| :--- | :--- |
| Prédictions | `[0.5, 0.5, 0.5, 0.5]` |
| **Loss BCE** | **0.6931** |

> [!WARNING]
> **Entropie Maximale :** Une perte de **0.6931** corresponds exactement à $-\ln(0.5)$. Cela signifie que le modèle est dans un état d'incertitude totale (50/50), ce qui est le point de départ classique avant l'entraînement.

---

## 🛠️ Environnement Technique
- **NumPy Version :** `1.24.3`
- **Fonction d'Activation :** Sigmoïde
- **Loss :** Binary Cross-Entropy (BCE)

---


---

# 📉 Rapport d'Expérimentation : Phase 2 - Apprentissage par Descente de Gradient

> [!IMPORTANT]
> **Objectif :** Automatiser la recherche des poids optimaux en implémentant l'algorithme de descente de gradient et la rétropropagation de l'erreur.

---

## 🔄 Processus d'Apprentissage
Le modèle n'est plus statique. Il ajuste ses paramètres à chaque itération (epoch) pour minimiser la fonction de perte.

```mermaid
graph TD
    A[Initialisation Aléatoire] --> B{Boucle d'Entraînement}
    B --> C[Forward Pass: Prédictions]
    C --> D[Calcul de la Loss: BCE]
    D --> E[Backward Pass: Calcul des Gradients]
    E --> F[Mise à jour: w = w - LR * dw]
    F --> B
    B -- Fin des Epochs --> G[Modèle Convergé]

    style A fill:#eee,stroke:#333
    style G fill:#bfb,stroke:#333
    style E fill:#fbb,stroke:#333
```

---

## 📈 Suivi de l'Entraînement
*Configuration : Learning Rate = 0.1 | Époques = 50*

| Époque | Loss (BCE) | Poids (w) | Biais (b) |
| :--- | :--- | :--- | :--- |
| 0 | **0.6934** | `[0.005, 0.015]` | `-0.000` |
| 10 | **0.6688** | `[-0.001, 0.170]` | `-0.010` |
| 20 | **0.6468** | `[-0.014, 0.316]` | `-0.032` |
| 30 | **0.6265** | `[-0.033, 0.453]` | `-0.063` |
| 40 | **0.6074** | `[-0.054, 0.585]` | `-0.098` |

---

## 🖼️ Courbe de Convergence

![Courbe de perte BCE](graph/phase2_loss_curve.png)

> [!TIP]
> **Analyse de la courbe :** On observe une diminution constante de la perte, ce qui confirme que le gradient calculé permet effectivement de descendre vers un minimum local. La pente s'adoucit au fil des itérations, signe que le modèle se rapproche de sa solution optimale pour ce taux d'apprentissage.

---

# 🌀 Rapport d'Expérimentation : Phase 3 - Le Problème du XOR

> [!IMPORTANT]
> **Objectif :** Résoudre le problème non-linéaire du XOR (eXclusive OR) en utilisant un Réseau de Neurones Multi-Couches (PMC). Cette phase démontre la nécessité d'au moins une couche cachée pour séparer des données non linéairement séparables.

---

## 🏗️ Architecture du Réseau (2-2-1)
Pour "plier" l'espace et isoler les classes du XOR, nous utilisons une couche cachée de 2 neurones.

```mermaid
graph LR
    subgraph Input
    X1((X1))
    X2((X2))
    end
    
    subgraph Hidden[Couche Cachée]
    H1((H1))
    H2((H2))
    end
    
    subgraph Output[Sortie]
    Out((Y))
    end
    
    X1 --> H1
    X1 --> H2
    X2 --> H1
    X2 --> H2
    H1 --> Out
    H2 --> Out
    
    style Hidden fill:#fdf,stroke:#333
    style Output fill:#bfb,stroke:#333
```

---

## 🔬 Analyse des Scénarios XOR

````carousel
### 🟢 Scénario 1 : Normal (Architecture 2-2-1)
*Le réseau réussit parfaitement à apprendre la logique XOR.*

| Époque | Loss | Accuracy |
| :--- | :--- | :--- |
| 0 | 0.6958 | 50.00% |
| 4000 | 0.0080 | 100.00% |
| 8000 | **0.0030** | **100.00%** |

![Frontière de décision Normal](graph/phase3_decision_boundary.png)

<!-- slide -->

### 🟡 Scénario 2 : Avec Bruit
*L'ajout de bruit teste la robustesse du modèle.*

| Époque | Loss | Accuracy |
| :--- | :--- | :--- |
| 0 | 0.6958 | 50.00% |
| 8000 | **0.0030** | **100.00%** |

![Frontière de décision Bruit](graph/phase3_decision_boundary_bruit.png)

<!-- slide -->

### 🔴 Scénario 3 : Échec (1 Neurone Caché)
*Preuve théorique : 1 seul neurone ne peut pas résoudre le XOR.*

| Époque | Loss | Accuracy |
| :--- | :--- | :--- |
| 0 | 0.6967 | 50.00% |
| 8000 | **0.6931** | **50.00%** |

![Frontière de décision Échec](graph/phase3_decision_boundary_un_neurone_caché.png)

> [!CAUTION]
> **Pourquoi l'échec ?** Avec un seul neurone caché, le réseau ne peut créer qu'une seule droite de séparation. Le XOR nécessite deux droites pour isoler les points (0,1) et (1,0) des points (0,0) et (1,1).
````

---
