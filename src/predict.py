import pandas as pd
from utils import get_dataset, load_model
from utils.data import save_dataframe

X, y = get_dataset()

model = load_model("src/artifacts/models/model.gzip")

prediction = model.predict(X)
df_prediction = pd.DataFrame(prediction, columns=["predictions"])

save_dataframe(df_prediction, "src/artifacts/datasets/predictions.csv")
print("Predictions Saved")
