from sklearn.model_selection import train_test_split
from utils import get_california_house_data
from utils.data import save_dataframe_as_csv
from utils.model import get_best_model_fitted, save_model,evaluation


def train_model():
    X, y = get_california_house_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

    for key, value in data.items():
        save_dataframe_as_csv(value, f"src/artifacts/data/{key}.csv")

    print(f"Train dataframe size: {X_train.shape}")
    print(f"Test dataframe size: {X_test.shape}")

    model = get_best_model_fitted(X_train, y_train)
    save_model(model, "src/artifacts/model/model.gzip")
    print("Model saved")

    # eval
    r2_score = evaluation(model, X_test, y_test)
    print(f"Rˆ2 score: {r2_score}")


if __name__ == "__main__":
    train_model()