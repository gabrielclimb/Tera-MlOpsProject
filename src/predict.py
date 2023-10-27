import pandas as pd

from utils.data import get_california_house_data, save_dataframe_as_csv
from utils.model import load_model

def predict() -> None:
    X, y = get_california_house_data()
    model = load_model("src/artifacts/model/model.gzip")

    predictions = model.predict(X)
    df_predictions = pd.DataFrame(predictions, columns=["prediction"])

    save_dataframe_as_csv(df_predictions, "src/artifacts/data/predictions.csv")
    print("predictions saved")

if __name__ == "__main__":
    predict()