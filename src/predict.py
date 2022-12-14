import pandas as pd

from utils.data import get_california_housing, save_dataframe
from utils.model import load_model

X, y = get_california_housing()

model = load_model("src/documents/model/model.gzip")

predictions = model.predict(X)

df_predictions = pd.DataFrame(predictions, columns=["predictions"])

save_dataframe(df_predictions, "src/documents/data/predictions.csv")
print("Predictions saved")
