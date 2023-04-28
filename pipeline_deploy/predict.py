import pandas as pd

from helpers.data import get_california_house_data, save_dataframe
from helpers.model import load_model

X, y = get_california_house_data()

model = load_model("pipeline_deploy/artifacts/model.gzip")

prediction = model.predict(X)
df_prediction = pd.DataFrame(prediction, columns=["prediction"])

save_dataframe(df_prediction, "pipeline_deploy/artifacts/predictions.csv")
print("Predictions Saved")
