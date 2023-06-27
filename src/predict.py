from datetime import datetime

import pandas as pd

from utils import get_california_house_data, save_dataframe
from utils.model import load_model

now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

X, y = get_california_house_data()
model = load_model("src/artifacts/model/model_2023-06-27T01:15:37Z.gzip")
prediction = model.predict(X)
print("prediction done")

df_prediction = pd.DataFrame(prediction, columns=["prediction"])
save_dataframe(df_prediction, f"src/artifacts/data/predictions_{now}.csv")
print("prediction saved")