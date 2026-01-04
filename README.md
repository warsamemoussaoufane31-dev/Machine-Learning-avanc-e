# California Housing Price Prediction

Projet de Machine Learning visant à prédire le prix médian des maisons en Californie à partir du dataset classique
"California Housing". Le projet suit un pipeline complet : récupération des données, exploration, préparation,
modélisation, évaluation, sauvegarde, et mise à disposition d'une API pour la prédiction.

## Changelog

Voir le résumé des changements récents et des tâches exécutées : [CHANGELOG.md](../CHANGELOG.md)

## 🚀 Fonctionnalités

- Téléchargement automatique du dataset depuis la source d'origine
- Découpage en jeu d'entraînement et de test de manière stratifiée
- Préparation des données avec `ColumnTransformer` et `Pipeline` (imputation, scaling, encodage)
- Entraînement de plusieurs modèles :
  - Régression linéaire
  - Arbre de décision
  - Forêt aléatoire
  - SVR (RBF)
  - Réseau de neurones MLP
- Évaluation par validation croisée (RMSE)
- Sauvegarde du pipeline et des résultats de modèles avec `joblib`
- API FastAPI pour faire des prédictions à partir de nouvelles données

```

🛠 Installation
Prérequis : Python 3.9+ recommandé.

👤 Auteur
Warsame Moussa Oufane et Nawel Mordi
