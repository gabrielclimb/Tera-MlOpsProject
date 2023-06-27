import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from .utils.model import load_model
from .utils.schemas import Data

model = load_model("app/artifacts/model/model_2023-06-27T01:15:37Z.gzip")

app = FastAPI()


@app.get("/")
def hello() -> str:
    return "Hello World, Tera API"


@app.post("/predict")
def predict(data: Data) -> dict:
    df = pd.DataFrame(jsonable_encoder(data), index=[0])
    prediction = model.predict(df)
    return {"prediction":prediction[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    