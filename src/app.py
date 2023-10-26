import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from .helpers.model import load_model
from .models import Data
from typing import Dict

model = load_model("artifacts/model_2023-08-17-02-03.gzip")

app = FastAPI()


@app.get("/")
def hello() -> str:
    return "Hello World, California"



@app.post("/predict")
def predict(data: Data) -> Dict[str, float]:
    '''
    Galera, a versão nova do pydantic e da fastAPI
    obriga a typar a chave e o valor do retorno quando é dict
    eu tinha colocado somente dict mas precisa ser Dict[str, float]
    '''
    df = pd.DataFrame(jsonable_encoder(data), index=[0])
    prediction = model.predict(df)
    return {"prediction": prediction[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
