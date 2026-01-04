# TP2 - Classification et Évaluation de Modèles

## 🎯 Objectifs du TP

Ce deuxième travail pratique vous permet d'explorer les problèmes de **classification** et d'approfondir l'évaluation des modèles de Machine Learning.

### Objectifs pédagogiques :
- Comprendre la différence entre régression et classification
- Maîtriser les algorithmes de classification (binaire et multiclasse)
- Utiliser des métriques d'évaluation adaptées à la classification
- Interpréter les matrices de confusion et courbes ROC
- Gérer les problèmes de classes déséquilibrées
- Optimiser les hyperparamètres des modèles

## 📊 Datasets suggérés

### 1. Iris Dataset (Classification multiclasse)
Dataset classique contenant 150 échantillons de fleurs d'iris avec 4 caractéristiques :
- Longueur du sépale
- Largeur du sépale
- Longueur du pétale
- Largeur du pétale
- **Classe cible** : 3 espèces (Setosa, Versicolor, Virginica)

### 2. Titanic Dataset (Classification binaire)
Prédiction de survie des passagers du Titanic :
- Caractéristiques : âge, sexe, classe, prix du billet, etc.
- **Classe cible** : survie (0 = décédé, 1 = survécu)

## 🚀 Fonctionnalités à implémenter

- Chargement et exploration des données
- Prétraitement adapté à la classification
- Gestion des classes déséquilibrées (SMOTE, class_weight)
- Entraînement de plusieurs classificateurs :
  - Régression logistique
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Arbre de décision
  - Forêt aléatoire
  - Gradient Boosting (XGBoost)
- Évaluation complète :
  - Matrice de confusion
  - Précision, Rappel, F1-Score
  - Courbes ROC et AUC
  - Courbes Precision-Recall
- Optimisation par Grid Search ou Random Search
- Validation croisée stratifiée

## 📁 Structure du projet

```text
TP2/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── .gitignore                        # Fichiers à ignorer par Git
├── notebooks/
│   ├── classification_iris.ipynb    # Notebook pour dataset Iris
│   └── classification_titanic.ipynb # Notebook pour dataset Titanic
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Chargement des données
│   ├── preprocessing.py             # Prétraitement des données
│   ├── models.py                    # Définition des modèles
│   ├── evaluation.py                # Métriques et évaluation
│   ├── visualization.py             # Visualisation des résultats
│   └── utils.py                     # Fonctions utilitaires
├── data/                            # Données (à télécharger)
└── models/                          # Modèles sauvegardés
```

## 🛠 Installation

### Prérequis
- Python 3.9+ recommandé
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
cd TP2
pip install -r requirements.txt
```

## 🎓 Instructions d'exécution

### Option 1 : Notebook Jupyter - Classification Iris

```bash
cd TP2
jupyter notebook notebooks/classification_iris.ipynb
```

Ce notebook couvre :
1. Chargement et exploration du dataset Iris
2. Visualisation des distributions de classes
3. Division train/test stratifiée
4. Entraînement de plusieurs classificateurs
5. Évaluation et comparaison des performances
6. Matrice de confusion et métriques détaillées

### Option 2 : Notebook Jupyter - Classification Titanic

```bash
cd TP2
jupyter notebook notebooks/classification_titanic.ipynb
```

Ce notebook explore :
1. Chargement et nettoyage des données Titanic
2. Feature engineering (création de nouvelles variables)
3. Gestion des valeurs manquantes
4. Encodage des variables catégorielles
5. Gestion des classes déséquilibrées
6. Optimisation des hyperparamètres
7. Analyse des features importantes

### Option 3 : Scripts Python

```bash
cd TP2

# 1. Charger les données
python -m src.data_loader --dataset iris

# 2. Prétraiter les données
python -m src.preprocessing

# 3. Entraîner les modèles
python -m src.models --train

# 4. Évaluer les modèles
python -m src.evaluation
```

## 📚 Concepts clés abordés

### 1. Types de classification
- **Classification binaire** : 2 classes (ex: spam/non-spam)
- **Classification multiclasse** : plus de 2 classes (ex: espèces d'iris)
- **Classification multi-label** : plusieurs étiquettes par échantillon

### 2. Algorithmes de classification
- **Régression logistique** : modèle linéaire avec fonction sigmoïde
- **K-NN** : classification basée sur les k voisins les plus proches
- **SVM** : recherche d'hyperplan optimal avec marge maximale
- **Arbres de décision** : règles de décision hiérarchiques
- **Forêt aléatoire** : ensemble d'arbres pour plus de robustesse
- **Gradient Boosting** : construction séquentielle de modèles

### 3. Métriques d'évaluation

#### Pour classification binaire :
- **Accuracy** : proportion de prédictions correctes
- **Precision** : proportion de vrais positifs parmi les prédictions positives
- **Recall (Sensibilité)** : proportion de vrais positifs détectés
- **F1-Score** : moyenne harmonique de Precision et Recall
- **AUC-ROC** : aire sous la courbe ROC
- **Matrice de confusion** : visualisation des TP, TN, FP, FN

#### Pour classification multiclasse :
- Métriques micro/macro/weighted average
- Confusion matrix multiclasse
- Classification report détaillé

### 4. Optimisation et validation
- **Validation croisée stratifiée** : préserver la distribution des classes
- **Grid Search** : recherche exhaustive d'hyperparamètres
- **Random Search** : recherche aléatoire plus efficace
- **Gestion du surapprentissage** : régularisation, early stopping

### 5. Problèmes spécifiques
- **Classes déséquilibrées** :
  - Ré-échantillonnage (SMOTE, RandomUnderSampler)
  - Ajustement des poids de classes (class_weight)
  - Métriques adaptées (F1-Score, AUC-ROC)
- **Feature importance** : identifier les variables les plus prédictives
- **Interprétabilité** : comprendre les décisions du modèle

## 🎯 Exercices suggérés

### Niveau débutant :
1. Implémenter une classification simple sur le dataset Iris
2. Calculer et interpréter la matrice de confusion
3. Comparer les performances de 3 algorithmes différents

### Niveau intermédiaire :
4. Appliquer la validation croisée stratifiée
5. Optimiser les hyperparamètres avec GridSearchCV
6. Tracer et interpréter les courbes ROC
7. Gérer les valeurs manquantes du dataset Titanic

### Niveau avancé :
8. Implémenter SMOTE pour gérer les classes déséquilibrées
9. Créer des features personnalisées (feature engineering)
10. Comparer micro/macro/weighted averaging pour les métriques
11. Analyser l'importance des features avec Random Forest
12. Implémenter un système d'ensemble (stacking, voting)

## 💡 Exemples de code

### Classification simple avec Scikit-learn

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Charger les données
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, stratify=iris.target, random_state=42
)

# Entraîner le modèle
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Prédire et évaluer
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### Optimisation des hyperparamètres

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Meilleurs paramètres : {grid_search.best_params_}")
```

## 📖 Ressources complémentaires

- [Scikit-learn Classification](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Understanding ROC Curves](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 👥 Auteurs

Warsame Moussa Oufane et Nawel Mordi

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](../LICENSE) à la racine du dépôt.
