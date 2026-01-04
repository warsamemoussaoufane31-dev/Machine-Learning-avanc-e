# TP5 - NLP et Traitement du Langage Naturel

## 🎯 Objectifs du TP

Ce cinquième travail pratique vous introduit au **Natural Language Processing (NLP)** et au traitement automatique du langage naturel. Vous apprendrez à traiter, analyser et modéliser des données textuelles.

### Objectifs pédagogiques :
- Comprendre les bases du prétraitement de texte
- Maîtriser les techniques de vectorisation (Bag of Words, TF-IDF, Word Embeddings)
- Implémenter des modèles de classification de texte
- Réaliser une analyse de sentiments
- Utiliser des modèles pré-entraînés (Word2Vec, BERT)
- Appliquer des réseaux de neurones récurrents (RNN, LSTM)
- Découvrir les transformers et l'attention

## 📊 Tâches et datasets

### 1. Analyse de sentiments
**Datasets** :
- **IMDB Reviews** : 50 000 critiques de films (positif/négatif)
- **Twitter Sentiment** : tweets classifiés par sentiment
- **Amazon Reviews** : avis clients sur produits

**Objectif** : Déterminer si un texte exprime un sentiment positif ou négatif

### 2. Classification de textes
**Datasets** :
- **20 Newsgroups** : articles de 20 catégories différentes
- **AG News** : articles de presse (4 catégories)
- **SMS Spam** : détection de spam dans les SMS

**Objectif** : Classer les textes dans des catégories prédéfinies

### 3. Named Entity Recognition (NER)
**Datasets** :
- **CoNLL-2003** : reconnaissance d'entités nommées
- Datasets français : FTB, WikiNER

**Objectif** : Identifier et classifier les entités (personnes, lieux, organisations)

### 4. Génération de texte
**Approches** :
- Modèles de langage basés sur caractères
- Modèles de langage basés sur mots
- Fine-tuning de GPT-2 ou T5

**Objectif** : Générer du texte cohérent

## 🚀 Techniques et méthodes

### 1. Prétraitement de texte

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    """Prétraitement complet d'un texte."""
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Suppression de la ponctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenisation
    tokens = text.split()
    
    # Suppression des stop words
    stop_words = set(stopwords.words('french'))
    tokens = [w for w in tokens if w not in stop_words]
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return ' '.join(tokens)
```

### 2. Vectorisation du texte

#### Bag of Words (BoW)
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus)
```

#### TF-IDF
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(corpus)
```

#### Word Embeddings (Word2Vec)
```python
from gensim.models import Word2Vec

# Entraîner Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)

# Obtenir le vecteur d'un mot
vector = model.wv['python']

# Similarité
similar = model.wv.most_similar('python', topn=10)
```

### 3. Modèles traditionnels

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Naive Bayes (rapide, bon pour texte)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# SVM
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
```

### 4. Deep Learning pour NLP

#### LSTM pour classification
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

#### CNN pour texte
```python
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### 5. Transfer Learning avec BERT

```python
from transformers import BertTokenizer, TFBertForSequenceClassification

# Charger le tokenizer et le modèle
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenisation
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')

# Entraînement
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(inputs, labels, epochs=3)
```

## 📁 Structure du projet

```text
TP5/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── .gitignore                        # Fichiers à ignorer par Git
├── notebooks/
│   ├── sentiment_analysis.ipynb     # Analyse de sentiments
│   ├── text_classification.ipynb    # Classification de textes
│   ├── word_embeddings.ipynb        # Word2Vec, GloVe
│   └── transformers_nlp.ipynb       # BERT, GPT
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # Prétraitement de texte
│   ├── vectorization.py             # Vectorisation (BoW, TF-IDF)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── traditional.py           # Modèles classiques (Naive Bayes, SVM)
│   │   ├── deep_learning.py         # LSTM, CNN pour texte
│   │   └── transformers.py          # BERT, RoBERTa
│   ├── embeddings.py                # Word2Vec, FastText
│   ├── evaluation.py                # Métriques et évaluation
│   ├── visualization.py             # Visualisation (word clouds, etc.)
│   └── utils.py                     # Fonctions utilitaires
├── data/                            # Données textuelles
└── models/                          # Modèles sauvegardés
    ├── embeddings/                  # Word embeddings
    └── classifiers/                 # Classificateurs
```

## 🛠 Installation

### Prérequis
- Python 3.9+ recommandé
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
cd TP5
pip install -r requirements.txt

# Télécharger les ressources NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Pour les modèles français
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### Installation de spaCy (optionnel)

```bash
pip install spacy
python -m spacy download fr_core_news_sm  # Français
python -m spacy download en_core_web_sm   # Anglais
```

## 🎓 Instructions d'exécution

### Notebook 1 : Analyse de sentiments (Débutant)

```bash
cd TP5
jupyter notebook notebooks/sentiment_analysis.ipynb
```

Ce notebook couvre :
1. Chargement d'un dataset de sentiments (IMDB ou Twitter)
2. Prétraitement du texte
3. Vectorisation avec TF-IDF
4. Entraînement de classificateurs (Naive Bayes, Logistic Regression)
5. Évaluation des performances
6. Analyse des erreurs

**Temps d'exécution** : ~10-15 minutes

### Notebook 2 : Classification de textes (Intermédiaire)

```bash
cd TP5
jupyter notebook notebooks/text_classification.ipynb
```

Ce notebook explore :
1. Dataset multiclasse (20 Newsgroups ou AG News)
2. Comparaison BoW vs TF-IDF
3. Feature engineering pour le texte
4. Optimisation des hyperparamètres
5. Visualisation avec word clouds
6. Interprétation des features importantes

**Temps d'exécution** : ~15-20 minutes

### Notebook 3 : Word Embeddings (Avancé)

```bash
cd TP5
jupyter notebook notebooks/word_embeddings.ipynb
```

Ce notebook présente :
1. Entraînement de Word2Vec sur un corpus personnalisé
2. Visualisation des embeddings avec t-SNE
3. Analogies de mots (roi - homme + femme = ?)
4. Utilisation de GloVe pré-entraîné
5. Application à la classification de texte

**Temps d'exécution** : ~20-30 minutes

### Notebook 4 : Transformers pour NLP (Expert)

```bash
cd TP5
jupyter notebook notebooks/transformers_nlp.ipynb
```

Ce notebook illustre :
1. Fine-tuning de BERT pour la classification
2. Utilisation de modèles multilingues
3. Transfer learning sur un dataset personnalisé
4. Comparaison des performances avec modèles classiques
5. Visualisation de l'attention

**Temps d'exécution** : ~30-60 minutes (GPU recommandé)

## 📚 Concepts clés abordés

### 1. Prétraitement NLP
- **Tokenisation** : découpage en mots/tokens
- **Normalisation** : minuscules, suppression accents
- **Stop words** : suppression des mots courants peu informatifs
- **Lemmatisation** : réduction aux formes canoniques (courir → courir)
- **Stemming** : réduction aux racines (courir → cour)
- **Nettoyage** : URLs, emails, ponctuation, chiffres

### 2. Représentations du texte

#### Approches traditionnelles
- **Bag of Words (BoW)** : comptage de mots
- **TF-IDF** : importance relative des termes
- **N-grams** : séquences de n mots consécutifs

#### Embeddings contextuels
- **Word2Vec** : représentation vectorielle des mots
- **GloVe** : factorisation de matrice de co-occurrence
- **FastText** : sous-mots pour mieux gérer les mots rares
- **ELMo** : embeddings contextuels
- **BERT** : bidirectionnel avec transformers

### 3. Modèles de classification

#### Classiques
- **Naive Bayes** : probabiliste, rapide, baseline solide
- **Logistic Regression** : linéaire, interprétable
- **SVM** : marge maximale, kernel tricks

#### Deep Learning
- **RNN** : réseaux récurrents pour séquences
- **LSTM/GRU** : gestion de la mémoire long terme
- **CNN 1D** : convolutions pour extraction de features
- **Attention** : mécanisme d'attention
- **Transformers** : architecture state-of-the-art (BERT, GPT, T5)

### 4. Métriques d'évaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Pour classification binaire
print(classification_report(y_true, y_pred))

# Pour multiclasse
print(classification_report(y_true, y_pred, target_names=categories))

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
```

### 5. Visualisation

#### Word Cloud
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

#### Visualisation des embeddings
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Réduire à 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(word_vectors)

# Visualiser
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
plt.show()
```

## 🎯 Exercices suggérés

### Niveau débutant :
1. Implémenter une analyse de sentiments sur IMDB
2. Comparer Naive Bayes et Logistic Regression
3. Créer un word cloud des mots les plus fréquents
4. Analyser l'impact de la suppression des stop words

### Niveau intermédiaire :
5. Classifier des articles de presse en catégories
6. Comparer BoW, TF-IDF et bi-grams
7. Entraîner un modèle Word2Vec sur un corpus français
8. Visualiser les word embeddings avec t-SNE
9. Implémenter un détecteur de spam pour emails

### Niveau avancé :
10. Construire un classificateur LSTM pour l'analyse de sentiments
11. Fine-tuner BERT sur un dataset personnalisé
12. Comparer les performances CNN vs LSTM vs BERT
13. Implémenter un système de questions-réponses simple
14. Créer un chatbot basé sur règles puis sur ML
15. Analyser l'attention dans un modèle Transformer

## 💡 Cas d'usage réels

### 1. Service client automatisé
- Classification des demandes clients
- Routage automatique vers le bon service
- Analyse de satisfaction

### 2. Modération de contenu
- Détection de contenu inapproprié
- Identification de spam
- Filtrage de toxicité

### 3. Veille et analyse
- Monitoring des réseaux sociaux
- Analyse de sentiments sur la marque
- Détection de tendances

### 4. Recommandation
- Recommandation d'articles similaires
- Matching de profils
- Suggestion de tags automatique

### 5. Extraction d'information
- Named Entity Recognition
- Extraction de relations
- Résumé automatique de texte

## 📖 Ressources complémentaires

### Livres et cours
- [Natural Language Processing with Python (NLTK Book)](https://www.nltk.org/book/)
- [Speech and Language Processing - Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/)
- [CS224N - Stanford NLP](http://web.stanford.edu/class/cs224n/)

### Documentation
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Gensim Documentation](https://radimrehurek.com/gensim/)

### Articles et tutoriels
- [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Text Classification Guide](https://developers.google.com/machine-learning/guides/text-classification)

## 👥 Auteurs

Warsame Moussa Oufane et Nawel Mordi

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](../LICENSE) à la racine du dépôt.
