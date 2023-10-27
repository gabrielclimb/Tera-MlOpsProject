from typing import Dict, List

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from .models.schemas import Data
from .utils.model import load_model

model = load_model("/code/app/model.gzip")

app = FastAPI()


@app.get("/")
def hello() -> str:
    return "Hello California"


@app.post("/predict")
def predict(data: Data) -> Dict[str, List[float]]:
    df = pd.DataFrame(jsonable_encoder(data), index=[0])
    prediction = model.predict(df)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
