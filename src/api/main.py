import os

import joblib
import pandas as pd
from fastapi import FastAPI

from .schemas import Data

app = FastAPI()

print(os.getcwd())
# Initialize files
model = joblib.load("src/model/regressor.gzip")


@app.post("/predict")
def predict(data: Data):
    """_summary_

    Args:
        data (Data): _description_

    Returns:
        _type_: _description_
    """

    # Extract data in correct order
    df = pd.DataFrame(data)

    # Create and return prediction
    prediction = model.predict(df)

    return {"prediction": prediction}
