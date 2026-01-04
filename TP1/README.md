# TP1 - Introduction au Machine Learning et Régression

## 🎯 Objectifs du TP

Ce premier travail pratique vous initie aux concepts fondamentaux du Machine Learning à travers un problème de **régression** : la prédiction des prix médians des maisons en Californie.

### Objectifs pédagogiques :
- Comprendre le pipeline complet d'un projet de Machine Learning
- Maîtriser les techniques de préparation des données
- Implémenter et comparer différents algorithmes de régression
- Évaluer les performances des modèles avec des métriques appropriées
- Déployer un modèle via une API REST

## 📊 Dataset : California Housing

Le dataset California Housing contient des informations sur les logements en Californie collectées lors du recensement de 1990. Il comprend :

- **8 caractéristiques** :
  - `longitude` et `latitude` : localisation géographique
  - `housing_median_age` : âge médian des maisons dans le secteur
  - `total_rooms` : nombre total de pièces
  - `total_bedrooms` : nombre total de chambres
  - `population` : population du secteur
  - `households` : nombre de ménages
  - `median_income` : revenu médian des ménages
  - `ocean_proximity` : proximité de l'océan (catégorielle)

- **Variable cible** : `median_house_value` (prix médian des maisons)

## 🚀 Fonctionnalités

- Téléchargement automatique du dataset depuis la source d'origine
- Découpage stratifié en jeu d'entraînement et de test
- Préparation des données avec `ColumnTransformer` et `Pipeline` (imputation, scaling, encodage)
- Entraînement de plusieurs modèles :
  - Régression linéaire
  - Arbre de décision
  - Forêt aléatoire
  - Support Vector Regression (SVR)
  - Réseau de neurones MLP
- Évaluation par validation croisée (RMSE)
- Sauvegarde du pipeline et des modèles avec `joblib`
- API FastAPI pour faire des prédictions en temps réel

## 📁 Structure du projet

```text
TP1/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── .gitignore                        # Fichiers à ignorer par Git
├── Dockerfile                        # Configuration Docker
├── docker-compose.yml                # Orchestration Docker
├── notebooks/
│   └── TP_Machine_learning.ipynb    # Notebook Jupyter du TP
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py              # Récupération des données
│   ├── data_exploration.py          # Exploration et visualisation
│   ├── data_preparation.py          # Préparation et transformation
│   ├── train_models.py              # Entraînement des modèles
│   ├── evaluate.py                  # Évaluation et métriques
│   ├── api.py                       # API FastAPI
│   └── utils.py                     # Fonctions utilitaires
├── data/                            # Données (générées automatiquement)
└── models/                          # Modèles sauvegardés
```

## 🛠 Installation

### Prérequis
- Python 3.9+ recommandé
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
cd TP1
pip install -r requirements.txt
```

### Avec Docker (optionnel)

```bash
cd TP1
docker-compose up
```

## 🎓 Instructions d'exécution

### Option 1 : Utilisation du Notebook Jupyter

Le notebook interactif est idéal pour l'apprentissage et l'expérimentation :

```bash
cd TP1
jupyter notebook notebooks/TP_Machine_learning.ipynb
```

Le notebook vous guide à travers toutes les étapes :
1. Récupération et exploration des données
2. Analyse exploratoire des données (EDA)
3. Préparation et transformation
4. Entraînement des modèles
5. Évaluation et comparaison
6. Sauvegarde des modèles

### Option 2 : Exécution des scripts Python

Vous pouvez aussi exécuter les scripts individuellement :

```bash
cd TP1

# 1. Récupérer les données
python -m src.data_fetcher

# 2. Explorer les données
python -m src.data_exploration

# 3. Préparer les données
python -m src.data_preparation

# 4. Entraîner les modèles
python -m src.train_models

# 5. Évaluer les modèles
python -m src.evaluate
```

### Option 3 : Lancer l'API de prédiction

Une fois les modèles entraînés, vous pouvez déployer l'API :

```bash
cd TP1
python -m src.api
```

L'API sera accessible à l'adresse : http://localhost:8000

Documentation de l'API : http://localhost:8000/docs

Exemple de requête :
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

## 📚 Concepts clés abordés

### 1. Préparation des données
- Gestion des valeurs manquantes (imputation)
- Normalisation et standardisation
- Encodage des variables catégorielles
- Feature engineering

### 2. Modèles de régression
- **Régression linéaire** : modèle de base simple et interprétable
- **Arbre de décision** : modèle non-linéaire, sensible au surapprentissage
- **Forêt aléatoire** : ensemble d'arbres pour plus de robustesse
- **SVR** : Support Vector Regression pour les relations complexes
- **MLP** : Réseau de neurones pour l'apprentissage de patterns complexes

### 3. Évaluation des modèles
- RMSE (Root Mean Squared Error)
- Validation croisée
- Comparaison des performances
- Analyse des résidus

### 4. Déploiement
- Sauvegarde et chargement de modèles
- Création d'une API REST
- Conteneurisation avec Docker

## 🎯 Exercices suggérés

1. **Améliorer les features** : Créez de nouvelles caractéristiques (ex: rooms_per_household, bedrooms_per_room)
2. **Optimiser les hyperparamètres** : Utilisez GridSearchCV pour trouver les meilleurs paramètres
3. **Tester d'autres modèles** : Essayez XGBoost, LightGBM ou CatBoost
4. **Analyser les erreurs** : Identifiez les cas où le modèle se trompe le plus
5. **Visualiser les prédictions** : Créez des graphiques comparant prédictions et valeurs réelles

## 📖 Ressources complémentaires

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)

## 👥 Auteurs

Warsame Moussa Oufane et Nawel Mordi

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](../LICENSE) à la racine du dépôt.
