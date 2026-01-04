# Machine Learning Avancé - Travaux Pratiques

Bienvenue dans le dépôt des travaux pratiques de Machine Learning Avancé ! Ce dépôt contient une série de 5 TP couvrant les concepts essentiels du Machine Learning, du Deep Learning et du NLP.

## 📚 Table des matières

- [À propos](#à-propos)
- [Structure du dépôt](#structure-du-dépôt)
- [Liste des TP](#liste-des-tp)
- [Prérequis généraux](#prérequis-généraux)
- [Installation](#installation)
- [Auteurs](#auteurs)
- [Licence](#licence)

## 🎯 À propos

Ce dépôt regroupe des travaux pratiques progressifs en Machine Learning, conçus pour vous accompagner de l'apprentissage des bases à la maîtrise de techniques avancées. Chaque TP est indépendant et peut être réalisé séparément.

Les TP sont organisés par thématique et difficulté croissante :
1. **Régression** : prédiction de valeurs continues
2. **Classification** : prédiction de classes discrètes
3. **Clustering** : découverte de structures sans supervision
4. **Deep Learning** : réseaux de neurones profonds
5. **NLP** : traitement du langage naturel

## 📁 Structure du dépôt

```text
Machine-Learning-avanc-e/
├── README.md                  # Ce fichier
├── LICENSE                   # Licence MIT
├── TP1/                      # Régression - California Housing
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   └── models/
├── TP2/                      # Classification - Iris & Titanic
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   └── models/
├── TP3/                      # Clustering - Segmentation
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   └── models/
├── TP4/                      # Deep Learning - CNN
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   └── models/
└── TP5/                      # NLP - Analyse de sentiments
    ├── README.md
    ├── requirements.txt
    ├── notebooks/
    ├── src/
    ├── data/
    └── models/
```

## 📖 Liste des TP

### [TP1 - Introduction au Machine Learning et Régression](./TP1/)

**Thématique** : Régression linéaire et prédiction de prix

**Objectifs** :
- Comprendre le pipeline complet d'un projet ML
- Maîtriser la préparation des données
- Comparer différents algorithmes de régression
- Déployer un modèle via une API REST

**Dataset** : California Housing (prix médians des maisons)

**Algorithmes** :
- Régression linéaire
- Arbre de décision
- Forêt aléatoire
- Support Vector Regression (SVR)
- Réseau de neurones MLP

**Technologies** : scikit-learn, pandas, FastAPI

**Niveau** : ⭐ Débutant

---

### [TP2 - Classification et Évaluation de Modèles](./TP2/)

**Thématique** : Classification binaire et multiclasse

**Objectifs** :
- Maîtriser les algorithmes de classification
- Comprendre les métriques d'évaluation (précision, rappel, F1-score)
- Interpréter les matrices de confusion et courbes ROC
- Gérer les classes déséquilibrées
- Optimiser les hyperparamètres

**Datasets** :
- Iris (classification multiclasse - 3 espèces de fleurs)
- Titanic (classification binaire - survie)

**Algorithmes** :
- Régression logistique
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Arbres de décision
- Forêt aléatoire
- Gradient Boosting (XGBoost)

**Technologies** : scikit-learn, imbalanced-learn, XGBoost

**Niveau** : ⭐⭐ Intermédiaire

---

### [TP3 - Clustering et Apprentissage Non-Supervisé](./TP3/)

**Thématique** : Découverte de structures cachées dans les données

**Objectifs** :
- Comprendre l'apprentissage non-supervisé
- Maîtriser les principaux algorithmes de clustering
- Choisir le bon algorithme selon les données
- Évaluer la qualité des clusters
- Réduire la dimensionnalité (PCA, t-SNE)

**Cas d'usage** :
- Segmentation de clients
- Clustering d'images (MNIST, Fashion-MNIST)
- Détection d'anomalies

**Algorithmes** :
- K-Means
- DBSCAN
- Hierarchical Clustering (CAH)
- Gaussian Mixture Models (GMM)
- Mean Shift

**Technologies** : scikit-learn, scipy, UMAP

**Niveau** : ⭐⭐ Intermédiaire

---

### [TP4 - Réseaux de Neurones et Deep Learning](./TP4/)

**Thématique** : Vision par ordinateur avec réseaux de neurones profonds

**Objectifs** :
- Comprendre l'architecture des réseaux de neurones
- Maîtriser TensorFlow/Keras
- Implémenter des CNN pour la classification d'images
- Utiliser des techniques de régularisation
- Appliquer le transfer learning

**Datasets** :
- MNIST (chiffres manuscrits)
- Fashion-MNIST (vêtements)
- CIFAR-10 (images couleur - 10 classes)
- CIFAR-100 (classification fine-grained)

**Architectures** :
- Perceptron Multi-Couches (MLP)
- Réseaux Convolutifs (CNN)
- CNN avancés avec Batch Normalization
- Transfer Learning (VGG16, ResNet, MobileNet)

**Technologies** : TensorFlow, Keras, PyTorch (optionnel)

**Niveau** : ⭐⭐⭐ Avancé

---

### [TP5 - NLP et Traitement du Langage Naturel](./TP5/)

**Thématique** : Traitement et analyse de données textuelles

**Objectifs** :
- Maîtriser le prétraitement de texte
- Comprendre les techniques de vectorisation
- Implémenter des modèles de classification de texte
- Utiliser des word embeddings (Word2Vec)
- Découvrir les transformers (BERT)

**Tâches** :
- Analyse de sentiments (IMDB, Twitter)
- Classification de textes (20 Newsgroups, AG News)
- Named Entity Recognition
- Génération de texte

**Techniques** :
- Bag of Words, TF-IDF
- Word2Vec, GloVe, FastText
- LSTM, GRU pour séquences
- Transformers (BERT, GPT)

**Technologies** : NLTK, spaCy, Gensim, Transformers (Hugging Face)

**Niveau** : ⭐⭐⭐ Avancé

---

## 🛠 Prérequis généraux

### Logiciels requis
- **Python** : version 3.9 ou supérieure recommandée
- **pip** : gestionnaire de paquets Python
- **Git** : pour cloner le dépôt
- **Jupyter** : pour exécuter les notebooks (optionnel mais recommandé)

### Connaissances préalables
- Bases de Python (variables, fonctions, boucles, listes)
- Notions de mathématiques (algèbre linéaire, statistiques de base)
- Utilisation basique de la ligne de commande

### Matériel recommandé
- **Pour TP1, TP2, TP3** : Ordinateur standard (CPU suffisant)
- **Pour TP4, TP5** : GPU recommandé mais pas obligatoire
  - Sans GPU : temps d'entraînement plus longs mais fonctionnel
  - Avec GPU : accélération significative (CUDA compatible)

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/warsamemoussaoufane31-dev/Machine-Learning-avanc-e.git
cd Machine-Learning-avanc-e
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Créer l'environnement
python -m venv venv

# Activer l'environnement
# Sur Windows :
venv\Scripts\activate
# Sur macOS/Linux :
source venv/bin/activate
```

### 3. Installer les dépendances d'un TP spécifique

Chaque TP a ses propres dépendances. Naviguez vers le dossier du TP et installez :

```bash
# Exemple pour TP1
cd TP1
pip install -r requirements.txt

# Ou pour TP4
cd TP4
pip install -r requirements.txt
```

### 4. Lancer Jupyter Notebook

```bash
# Depuis le dossier d'un TP
jupyter notebook
```

Puis ouvrez le notebook souhaité dans votre navigateur.

## 📝 Guide d'utilisation

### Approche recommandée

1. **Commencez par TP1** si vous débutez en ML
2. **Suivez l'ordre des TP** pour une progression logique
3. **Lisez le README de chaque TP** avant de commencer
4. **Expérimentez** : modifiez les paramètres, testez de nouvelles idées
5. **Faites les exercices suggérés** pour approfondir

### Chaque TP est indépendant

- Vous pouvez faire les TP dans l'ordre de votre choix
- Chaque TP a ses propres dépendances
- Les données sont téléchargées automatiquement ou fournies

### Organisation du travail

Chaque TP contient :
- **README.md** : documentation complète du TP
- **notebooks/** : notebooks Jupyter interactifs
- **src/** : code Python réutilisable
- **data/** : données (générées automatiquement)
- **models/** : modèles entraînés sauvegardés
- **requirements.txt** : dépendances Python

## 🎓 Ressources complémentaires

### Cours en ligne
- [Machine Learning - Coursera (Andrew Ng)](https://www.coursera.org/learn/machine-learning)
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai - Practical Deep Learning](https://www.fast.ai/)
- [CS229 - Stanford Machine Learning](http://cs229.stanford.edu/)

### Livres
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" - Aurélien Géron
- "Deep Learning" - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Pattern Recognition and Machine Learning" - Christopher Bishop

### Documentation
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [Keras](https://keras.io/)

### Communautés
- [Kaggle](https://www.kaggle.com/) - Compétitions et datasets
- [Papers With Code](https://paperswithcode.com/) - Publications avec implémentations
- [Hugging Face](https://huggingface.co/) - Modèles pré-entraînés

## 🤝 Contribution

Les contributions sont les bienvenues ! Si vous souhaitez :
- Corriger des bugs
- Améliorer la documentation
- Ajouter des exemples
- Proposer de nouveaux TP

N'hésitez pas à ouvrir une issue ou une pull request.

## 📧 Support

Si vous rencontrez des difficultés :
1. Consultez le README du TP concerné
2. Vérifiez que toutes les dépendances sont installées
3. Recherchez l'erreur sur Google/Stack Overflow
4. Ouvrez une issue sur GitHub avec une description détaillée

## 👥 Auteurs

**Warsame Moussa Oufane** et **Nawel Mordi**

Étudiants en Machine Learning Avancé

## 📝 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🌟 Remerciements

Merci d'utiliser ce dépôt pour votre apprentissage du Machine Learning ! N'hésitez pas à mettre une étoile ⭐ si vous trouvez ce contenu utile.

**Bon apprentissage ! 🚀**
