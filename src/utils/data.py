from sklearn.datasets import fetch_california_housing
import pandas as pd


def get_dataset() -> tuple:
    X_house, y_house = fetch_california_housing(return_X_y=True, as_frame=True)
    X_house["AvgBedsPerRoom"] = X_house["AveBedrms"] / X_house["AveRooms"]
    return X_house, y_house


def save_dataframe(dataframe: pd.DataFrame, path: str) -> None:
    dataframe.to_csv(path)
