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

## 📁 Structure du projet

```text
california-housing-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── data_fetcher.py
│   ├── data_exploration.py
│   ├── data_preparation.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── api.py
│   └── utils.py
├── notebooks/
│   └── TP1_ML_California_Housing.ipynb
├── models/
└── data/
```

### src/ : scripts Python modulaires

### notebooks/ : notebook Jupyter d’origine

### models/ : fichiers de modèles sauvegardés (joblib)

### data/ : dataset téléchargé automatiquement

🛠 Installation
Prérequis : Python 3.9+ recommandé.

```bash
git clone https://github.com/<ton-compte>/california-housing-ml.git
cd california-housing-ml
python -m venv .venv
source .venv/bin/activate   # sur Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Uploader un notebook depuis ta machine locale (optionnel)

Si tu as un `.ipynb` local et veux l'ajouter au dépôt sans l'ouvrir dans l'interface Web, utilise le script `tools/upload_notebook.py` depuis la racine du repo :

```bash
# Exemple :
python tools/upload_notebook.py /chemin/vers/TP1_ML_California_Housing.ipynb --name TP1_ML_California_Housing.ipynb
```

Le script copie le notebook dans `california-housing-ml/notebooks/`, commit et pousse sur `origin/main`.

▶️ Entraînement des modèles

```bash
python -m src.train_models
```

▶️ Évaluation

```bash
python -m src.evaluate
```

🌐 Lancer l’API de prédiction

```bash
uvicorn src.api:app --reload
# http://127.0.0.1:8000/docs
```

🐳 Docker

```bash
docker build -t california-housing-ml .
docker run -p 8000:8000 california-housing-ml
```

👤 Auteur
Warsame Moussa Oufane et Nawel Mordi
