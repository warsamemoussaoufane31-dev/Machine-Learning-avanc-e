import os
import pandas as pd
import matplotlib.pyplot as plt


def load_housing_data(path: str = "data/housing/housing.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Run data_fetcher.fetch_housing_data() first.")
    return pd.read_csv(path)


def explore_data(df: pd.DataFrame) -> None:
    print("\nHead:\n", df.head())
    print("\nInfo:\n")
    print(df.info())
    print("\nDescribe:\n", df.describe())
    print("\nValue counts for 'ocean_proximity':\n", df["ocean_proximity"].value_counts())


def plot_histograms(df: pd.DataFrame) -> None:
    df.hist(bins=50, figsize=(20, 15))
    plt.show()
