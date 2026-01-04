# TP3 - Clustering et Apprentissage Non-Supervisé

## 🎯 Objectifs du TP

Ce troisième travail pratique explore l'**apprentissage non-supervisé** à travers les algorithmes de clustering. Contrairement aux TP précédents, ici les données ne sont pas étiquetées : l'objectif est de découvrir des structures cachées dans les données.

### Objectifs pédagogiques :
- Comprendre les différences entre apprentissage supervisé et non-supervisé
- Maîtriser les principaux algorithmes de clustering
- Choisir le bon algorithme selon le type de données
- Évaluer la qualité des clusters
- Appliquer le clustering à des cas d'usage réels
- Réduire la dimensionnalité des données (PCA, t-SNE)

## 📊 Cas d'usage et datasets

### 1. Segmentation de clients
Regrouper les clients selon leurs comportements d'achat :
- Caractéristiques : montant dépensé, fréquence d'achat, récence, etc.
- Objectif : identifier des segments de clientèle pour le marketing ciblé

### 2. Clustering d'images
Regrouper des images similaires :
- Dataset MNIST : chiffres manuscrits
- Dataset Fashion-MNIST : vêtements et accessoires
- Objectif : découvrir des groupes naturels d'images

### 3. Détection d'anomalies
Identifier des points aberrants dans les données :
- Transactions frauduleuses
- Défauts de fabrication
- Comportements anormaux

## 🚀 Algorithmes implémentés

### 1. K-Means
- **Type** : Partitionnement
- **Principe** : Minimiser la distance intra-cluster
- **Avantages** : Simple, rapide, scalable
- **Inconvénients** : Nécessite de spécifier k, sensible aux outliers
- **Usage** : Segmentation client, compression d'images

### 2. DBSCAN (Density-Based Spatial Clustering)
- **Type** : Basé sur la densité
- **Principe** : Regrouper les points denses, identifier les outliers
- **Avantages** : Détecte les formes arbitraires, identifie le bruit
- **Inconvénients** : Sensible aux paramètres eps et min_samples
- **Usage** : Détection d'anomalies, clustering géospatial

### 3. Hierarchical Clustering (CAH)
- **Type** : Hiérarchique
- **Principe** : Construire une hiérarchie de clusters
- **Avantages** : Pas besoin de spécifier k, dendrogramme informatif
- **Inconvénients** : Complexité O(n³), pas scalable
- **Usage** : Taxonomie, analyse exploratoire

### 4. Gaussian Mixture Models (GMM)
- **Type** : Probabiliste
- **Principe** : Modéliser les données comme un mélange de gaussiennes
- **Avantages** : Appartenance probabiliste, flexible
- **Inconvénients** : Plus complexe, peut converger vers un minimum local
- **Usage** : Soft clustering, modélisation de densité

### 5. Mean Shift
- **Type** : Basé sur la densité
- **Principe** : Trouver les modes de la distribution de densité
- **Avantages** : Pas besoin de spécifier k, robuste
- **Inconvénients** : Lent, sensible au bandwidth
- **Usage** : Suivi d'objets, segmentation d'images

## 📁 Structure du projet

```text
TP3/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── .gitignore                        # Fichiers à ignorer par Git
├── notebooks/
│   ├── clustering_clients.ipynb     # Segmentation de clients
│   ├── clustering_images.ipynb      # Clustering d'images
│   └── detection_anomalies.ipynb    # Détection d'outliers
├── src/
│   ├── __init__.py
│   ├── data_generator.py            # Génération de données synthétiques
│   ├── clustering_models.py         # Implémentation des algorithmes
│   ├── evaluation.py                # Métriques d'évaluation
│   ├── visualization.py             # Visualisation des clusters
│   ├── dimensionality_reduction.py  # PCA, t-SNE, UMAP
│   └── utils.py                     # Fonctions utilitaires
├── data/                            # Données (générées ou téléchargées)
└── models/                          # Modèles sauvegardés
```

## 🛠 Installation

### Prérequis
- Python 3.9+ recommandé
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
cd TP3
pip install -r requirements.txt
```

## 🎓 Instructions d'exécution

### Notebook 1 : Segmentation de clients

```bash
cd TP3
jupyter notebook notebooks/clustering_clients.ipynb
```

Ce notebook couvre :
1. Génération ou chargement de données clients
2. Analyse exploratoire et préparation
3. Détermination du nombre optimal de clusters (méthode du coude, silhouette)
4. Application de K-Means
5. Profilage des segments identifiés
6. Visualisation avec PCA

### Notebook 2 : Clustering d'images

```bash
cd TP3
jupyter notebook notebooks/clustering_images.ipynb
```

Ce notebook explore :
1. Chargement du dataset MNIST ou Fashion-MNIST
2. Réduction de dimensionnalité (PCA)
3. Application de différents algorithmes de clustering
4. Visualisation avec t-SNE
5. Évaluation de la qualité des clusters

### Notebook 3 : Détection d'anomalies

```bash
cd TP3
jupyter notebook notebooks/detection_anomalies.ipynb
```

Ce notebook présente :
1. Génération de données avec outliers
2. Application de DBSCAN pour détecter les anomalies
3. Comparaison avec Isolation Forest
4. Visualisation des outliers détectés

## 📚 Concepts clés abordés

### 1. Choix du nombre de clusters

#### Méthode du coude (Elbow method)
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie')
plt.title('Méthode du coude')
plt.show()
```

#### Score de silhouette
```python
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"k={k}: silhouette={score:.3f}")
```

### 2. Métriques d'évaluation

- **Silhouette Score** : mesure de la cohésion et séparation (-1 à 1)
- **Davies-Bouldin Index** : rapport de similarité intra/inter-cluster (plus bas = mieux)
- **Calinski-Harabasz Index** : rapport de variance inter/intra-cluster (plus haut = mieux)
- **Adjusted Rand Index (ARI)** : si labels vrais disponibles

### 3. Réduction de dimensionnalité

#### PCA (Principal Component Analysis)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Variance expliquée : {pca.explained_variance_ratio_}")
```

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
```

#### UMAP (Uniform Manifold Approximation and Projection)
```python
import umap

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)
```

### 4. Cas particuliers

#### Gestion des données hétérogènes
- K-Prototypes pour données mixtes (numériques + catégorielles)
- Normalisation et standardisation essentielles

#### Clustering hiérarchique
```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.title('Dendrogramme')
plt.show()
```

## 🎯 Exercices suggérés

### Niveau débutant :
1. Appliquer K-Means sur un dataset simple (Iris)
2. Utiliser la méthode du coude pour choisir k
3. Visualiser les clusters avec PCA

### Niveau intermédiaire :
4. Comparer K-Means et DBSCAN sur le même dataset
5. Calculer et interpréter le silhouette score
6. Créer un dendrogramme avec clustering hiérarchique
7. Appliquer t-SNE pour visualiser des données haute dimension

### Niveau avancé :
8. Implémenter une segmentation client complète (RFM)
9. Utiliser GMM pour un clustering probabiliste
10. Détecter des anomalies avec DBSCAN et Isolation Forest
11. Optimiser les paramètres de DBSCAN (eps, min_samples)
12. Créer un pipeline de clustering avec prétraitement

## 💡 Applications pratiques

### 1. Marketing et CRM
- Segmentation de clientèle
- Personnalisation des offres
- Analyse de la churn

### 2. Traitement d'images
- Compression d'images
- Segmentation d'images médicales
- Reconnaissance de patterns

### 3. Bioinformatique
- Classification de gènes
- Identification de sous-types de maladies
- Analyse de séquences

### 4. Détection de fraude
- Identification de transactions suspectes
- Détection d'intrusions réseau
- Anomalies dans les logs

### 5. Recommandation
- Regroupement d'utilisateurs similaires
- Recommandation collaborative
- Découverte de niches

## 📖 Ressources complémentaires

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [DBSCAN Explained](https://towardsdatascience.com/dbscan-algorithm-complete-guide-and-application-with-python-scikit-learn-d690cbae4c5d)
- [K-Means Clustering](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)
- [t-SNE Visualization](https://distill.pub/2016/misread-tsne/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)

## 👥 Auteurs

Warsame Moussa Oufane et Nawel Mordi

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](../LICENSE) à la racine du dépôt.
