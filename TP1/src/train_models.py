import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

from .data_fetcher import fetch_housing_data
from .data_exploration import load_housing_data
from .data_preparation import stratified_split, build_pipeline
from .utils import display_scores


def main() -> None:
    os.makedirs("models", exist_ok=True)

    fetch_housing_data()
    df = load_housing_data()

    train_set, test_set = stratified_split(df)

    housing = train_set.drop("median_house_value", axis=1)
    labels = train_set["median_house_value"].copy()

    pipeline = build_pipeline(housing)
    housing_prepared = pipeline.fit_transform(housing)

    models = {
        "lin_reg": LinearRegression(),
        "tree_reg": DecisionTreeRegressor(random_state=42),
        "forest_reg": RandomForestRegressor(random_state=42),
        "svr_rbf": SVR(kernel="rbf", C=10, epsilon=0.2),
        "mlp_reg": MLPRegressor(random_state=42, max_iter=500)
    }

    results = {}
    best_model_name = None
    best_score = np.inf

    for name, model in models.items():
        scores = cross_val_score(
            model,
            housing_prepared,
            labels,
            scoring="neg_mean_squared_error",
            cv=5
        )
        rmse_scores = np.sqrt(-scores)
        print(f"\n{name} results:")
        display_scores(rmse_scores)
        mean_rmse = rmse_scores.mean()
        results[name] = {
            "rmse_mean": float(mean_rmse),
            "rmse_std": float(rmse_scores.std())
        }
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_model_name = name

    best_model = models[best_model_name]
    best_model.fit(housing_prepared, labels)

    artifact = {
        "pipeline": pipeline,
        "model": best_model,
        "best_model_name": best_model_name,
        "cv_results": results
    }

    output_path = "models/first_pass_models.joblib"
    joblib.dump(artifact, output_path)
    print(f"\nSaved best model '{best_model_name}' to {output_path} with RMSE={best_score:.2f}")


if __name__ == "__main__":
    main()
