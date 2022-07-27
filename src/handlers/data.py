import pandas as pd
from sklearn.datasets import fetch_california_housing


def get_california_house_data() -> tuple:
    X_house, y_house = fetch_california_housing(return_X_y=True, as_frame=True)
    X_house["AvgBedsPerRoom"] = X_house["AveBedrms"] / X_house["AveRooms"]
    return X_house, y_house


def save_dataframe(dataframe: pd.DataFrame, path: str) -> None:
    dataframe.to_csv(path)
