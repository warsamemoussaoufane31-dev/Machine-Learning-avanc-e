# TP4 - Réseaux de Neurones et Deep Learning

## 🎯 Objectifs du TP

Ce quatrième travail pratique vous introduit au **Deep Learning** et aux réseaux de neurones profonds. Vous apprendrez à construire, entraîner et optimiser des architectures neuronales pour diverses tâches.

### Objectifs pédagogiques :
- Comprendre l'architecture des réseaux de neurones
- Maîtriser les frameworks TensorFlow/Keras et PyTorch
- Implémenter des CNN pour la vision par ordinateur
- Utiliser des techniques de régularisation (Dropout, Batch Normalization)
- Optimiser l'entraînement (learning rate, optimizers, callbacks)
- Appliquer le transfer learning
- Comprendre les concepts de surapprentissage et early stopping

## 📊 Datasets et tâches

### 1. MNIST - Chiffres manuscrits
- **Description** : 70 000 images 28x28 en niveaux de gris (0-9)
- **Tâche** : Classification multiclasse (10 classes)
- **Niveau** : Débutant
- **Architecture** : MLP ou CNN simple

### 2. Fashion-MNIST - Articles de mode
- **Description** : 70 000 images 28x28 (vêtements et accessoires)
- **Tâche** : Classification multiclasse (10 classes)
- **Niveau** : Débutant/Intermédiaire
- **Architecture** : CNN

### 3. CIFAR-10 - Images couleur
- **Description** : 60 000 images 32x32 en couleur (10 classes)
- **Tâche** : Classification d'objets
- **Niveau** : Intermédiaire
- **Architecture** : CNN profond, ResNet, VGG

### 4. CIFAR-100 - Classification fine
- **Description** : 60 000 images 32x32 en couleur (100 classes)
- **Tâche** : Classification fine-grained
- **Niveau** : Avancé
- **Architecture** : Transfer learning, architectures pré-entraînées

## 🚀 Architectures implémentées

### 1. Perceptron Multi-Couches (MLP)
```python
# Architecture basique
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```

### 2. Réseau de Neurones Convolutif (CNN)
```python
# CNN pour images
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

### 3. CNN Avancé avec Batch Normalization
```python
# Architecture améliorée
model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(32, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),
    
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

### 4. Transfer Learning avec modèles pré-entraînés
```python
# Utiliser VGG16, ResNet, ou MobileNet
from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Geler les poids

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

## 📁 Structure du projet

```text
TP4/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── .gitignore                        # Fichiers à ignorer par Git
├── notebooks/
│   ├── mnist_mlp.ipynb              # MLP sur MNIST
│   ├── fashion_mnist_cnn.ipynb      # CNN sur Fashion-MNIST
│   ├── cifar10_cnn.ipynb            # CNN profond sur CIFAR-10
│   └── transfer_learning.ipynb      # Transfer learning
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Chargement et préparation des données
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py                   # Architectures MLP
│   │   ├── cnn.py                   # Architectures CNN
│   │   └── pretrained.py            # Modèles pré-entraînés
│   ├── training.py                  # Fonctions d'entraînement
│   ├── evaluation.py                # Évaluation et métriques
│   ├── visualization.py             # Visualisation des résultats
│   ├── callbacks.py                 # Callbacks personnalisés
│   └── utils.py                     # Fonctions utilitaires
├── data/                            # Données (téléchargées automatiquement)
└── models/                          # Modèles sauvegardés
    ├── checkpoints/                 # Checkpoints d'entraînement
    └── final/                       # Modèles finaux
```

## 🛠 Installation

### Prérequis
- Python 3.9+ recommandé
- GPU recommandé mais pas obligatoire
- CUDA et cuDNN (pour utilisation GPU avec TensorFlow)

### Installation des dépendances

#### Option 1 : TensorFlow/Keras (recommandé pour débutants)
```bash
cd TP4
pip install -r requirements.txt
```

#### Option 2 : PyTorch (alternative)
```bash
cd TP4
pip install -r requirements_pytorch.txt
```

### Vérification de l'installation

```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

## 🎓 Instructions d'exécution

### Notebook 1 : MLP sur MNIST (Débutant)

```bash
cd TP4
jupyter notebook notebooks/mnist_mlp.ipynb
```

Ce notebook couvre :
1. Chargement et exploration du dataset MNIST
2. Normalisation des données
3. Construction d'un MLP simple
4. Entraînement et évaluation
5. Visualisation des prédictions et erreurs
6. Analyse de la matrice de confusion

**Temps d'entraînement** : ~5 minutes (CPU)

### Notebook 2 : CNN sur Fashion-MNIST (Intermédiaire)

```bash
cd TP4
jupyter notebook notebooks/fashion_mnist_cnn.ipynb
```

Ce notebook explore :
1. Chargement de Fashion-MNIST
2. Construction d'un CNN
3. Techniques de régularisation (Dropout)
4. Callbacks (EarlyStopping, ModelCheckpoint)
5. Visualisation des filtres convolutifs
6. Analyse des erreurs de classification

**Temps d'entraînement** : ~10-15 minutes (CPU)

### Notebook 3 : CNN profond sur CIFAR-10 (Avancé)

```bash
cd TP4
jupyter notebook notebooks/cifar10_cnn.ipynb
```

Ce notebook présente :
1. Chargement et augmentation des données CIFAR-10
2. Construction d'un CNN profond
3. Batch Normalization
4. Learning rate scheduling
5. Data augmentation
6. Comparaison de différentes architectures

**Temps d'entraînement** : ~30-60 minutes (CPU), ~10 minutes (GPU)

### Notebook 4 : Transfer Learning (Expert)

```bash
cd TP4
jupyter notebook notebooks/transfer_learning.ipynb
```

Ce notebook illustre :
1. Utilisation de modèles pré-entraînés (VGG16, ResNet50, MobileNet)
2. Fine-tuning de couches spécifiques
3. Feature extraction
4. Comparaison des performances
5. Application à un dataset personnalisé

**Temps d'entraînement** : Variable selon le fine-tuning

## 📚 Concepts clés abordés

### 1. Composants d'un réseau de neurones

#### Couches de base
- **Dense (Fully Connected)** : connexions complètes entre neurones
- **Conv2D** : convolution 2D pour extraction de features
- **MaxPooling2D / AveragePooling2D** : réduction de dimensionnalité
- **Dropout** : régularisation par désactivation aléatoire
- **BatchNormalization** : normalisation par batch

#### Fonctions d'activation
- **ReLU** : f(x) = max(0, x) - standard pour couches cachées
- **Sigmoid** : f(x) = 1/(1+e^-x) - classification binaire
- **Softmax** : classification multiclasse
- **Tanh** : f(x) = (e^x - e^-x)/(e^x + e^-x)
- **LeakyReLU, ELU** : variantes de ReLU

### 2. Optimisation

#### Optimizers
```python
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Adam : adaptatif, bon choix par défaut
optimizer = Adam(learning_rate=0.001)

# SGD avec momentum
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# RMSprop : bon pour RNN
optimizer = RMSprop(learning_rate=0.001)
```

#### Learning Rate Scheduling
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)
```

### 3. Régularisation

#### Techniques principales
- **Dropout** : désactiver aléatoirement des neurones
- **L1/L2 Regularization** : pénaliser les poids élevés
- **Batch Normalization** : normaliser les activations
- **Data Augmentation** : augmenter artificiellement les données
- **Early Stopping** : arrêter l'entraînement avant surapprentissage

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

### 4. Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Entraînement avec augmentation
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=50,
          validation_data=(X_val, y_val))
```

### 5. Visualisation et interprétation

#### Visualisation des filtres
```python
import matplotlib.pyplot as plt

# Obtenir les poids de la première couche convolutive
filters = model.layers[0].get_weights()[0]

# Visualiser
fig, axes = plt.subplots(4, 8, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    if i < filters.shape[3]:
        ax.imshow(filters[:,:,0,i], cmap='gray')
        ax.axis('off')
plt.show()
```

#### Feature Maps
```python
from tensorflow.keras.models import Model

# Créer un modèle intermédiaire
layer_output = model.get_layer('conv2d_1').output
intermediate_model = Model(inputs=model.input, outputs=layer_output)

# Obtenir les feature maps
feature_maps = intermediate_model.predict(X_test[0:1])
```

## 🎯 Exercices suggérés

### Niveau débutant :
1. Construire et entraîner un MLP sur MNIST
2. Comparer les performances avec différents nombres de couches
3. Tester différentes fonctions d'activation
4. Implémenter un callback pour sauvegarder le meilleur modèle

### Niveau intermédiaire :
5. Construire un CNN pour Fashion-MNIST
6. Appliquer la data augmentation
7. Comparer CNN vs MLP sur le même dataset
8. Analyser l'impact du Dropout et de la Batch Normalization
9. Implémenter un learning rate scheduler

### Niveau avancé :
10. Construire un CNN profond pour CIFAR-10
11. Implémenter une architecture ResNet simplifiée
12. Appliquer le transfer learning avec VGG16 ou ResNet50
13. Fine-tuner les dernières couches d'un modèle pré-entraîné
14. Comparer les performances de différentes architectures
15. Implémenter Grad-CAM pour visualiser les zones importantes

## 💡 Bonnes pratiques

### 1. Préparation des données
- Normaliser les pixels entre 0 et 1 : `X = X / 255.0`
- Utiliser `to_categorical()` pour one-hot encoding
- Séparer train/val/test (60-20-20 typique)

### 2. Architecture
- Commencer simple, puis complexifier
- Augmenter progressivement le nombre de filtres
- Utiliser Dropout après les couches Dense
- Batch Normalization avant ou après activation

### 3. Entraînement
- Utiliser Adam comme optimizer par défaut
- Batch size : 32 ou 64 (selon GPU)
- Monitorer train et validation loss
- Sauvegarder les checkpoints

### 4. Éviter le surapprentissage
- Early stopping
- Data augmentation
- Dropout et régularisation L2
- Réduire la capacité du modèle si nécessaire

## 📖 Ressources complémentaires

### Documentation
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

### Tutoriels
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [CS231n - Stanford](http://cs231n.stanford.edu/)
- [Fast.ai Course](https://www.fast.ai/)

### Papers fondateurs
- [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [Very Deep CNNs (VGG)](https://arxiv.org/abs/1409.1556)
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)

## 👥 Auteurs

Warsame Moussa Oufane et Nawel Mordi

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](../LICENSE) à la racine du dépôt.
