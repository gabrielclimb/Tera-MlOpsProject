import pandas as pd

from handlers.data import get_california_house_data, save_dataframe
from handlers.model import load_model

X_all, y_all = get_california_house_data()

model = load_model("src/artifacts/model/model.gzip")

prediction = model.predict(X_all)
df_prediction = pd.DataFrame(prediction, columns=["prediction"])

save_dataframe(df_prediction, "src/artifacts/data/predictions.csv")
print(f"Prediction saved")
