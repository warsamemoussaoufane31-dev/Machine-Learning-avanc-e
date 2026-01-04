"""
Chargement des datasets de classification.
"""

from sklearn.datasets import load_iris, fetch_openml
import pandas as pd
import os


def charger_iris():
    """
    Charge le dataset Iris depuis scikit-learn.
    
    Returns:
        tuple: (X, y, feature_names, target_names)
    """
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names


def charger_titanic():
    """
    Charge le dataset Titanic depuis OpenML.
    
    Returns:
        tuple: (X, y) où X est un DataFrame et y est la cible
    """
    try:
        titanic = fetch_openml('titanic', version=1, as_frame=True)
        X = titanic.data
        y = titanic.target
        return X, y
    except Exception as e:
        print(f"Erreur lors du chargement du dataset Titanic : {e}")
        print("Utilisation d'un fichier local si disponible...")
        
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'titanic.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            # Supposer que 'Survived' est la colonne cible
            if 'Survived' in df.columns:
                y = df['Survived']
                X = df.drop('Survived', axis=1)
                return X, y
        
        raise FileNotFoundError("Dataset Titanic non disponible")


if __name__ == "__main__":
    # Test de chargement
    print("Chargement du dataset Iris...")
    X_iris, y_iris, features, targets = charger_iris()
    print(f"Iris chargé : {X_iris.shape[0]} échantillons, {X_iris.shape[1]} features")
    print(f"Classes : {targets}")
    
    print("\nChargement du dataset Titanic...")
    try:
        X_titanic, y_titanic = charger_titanic()
        print(f"Titanic chargé : {X_titanic.shape[0]} échantillons, {X_titanic.shape[1]} features")
    except Exception as e:
        print(f"Impossible de charger Titanic : {e}")
