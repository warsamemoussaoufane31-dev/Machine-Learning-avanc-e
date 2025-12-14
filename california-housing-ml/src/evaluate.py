import joblib
import numpy as np

from .data_exploration import load_housing_data
from .data_preparation import stratified_split


def main() -> None:
    artifact_path = "models/first_pass_models.joblib"
    artifact = joblib.load(artifact_path)

    pipeline = artifact["pipeline"]
    model = artifact["model"]

    df = load_housing_data()
    train_set, test_set = stratified_split(df)

    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()

    X_test_prepared = pipeline.transform(X_test)
    predictions = model.predict(X_test_prepared)

    rmse = np.sqrt(((predictions - y_test) ** 2).mean())
    print(f"Test RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()
