import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from .schemas import Data

# Initialize files
model = joblib.load("src/model/regressor.gzip")

app = FastAPI()


@app.get("/")
def hello() -> str:
    """just a standard route

    Returns:
        str: hello world
    """
    return "Hello World, go to /docs"


@app.post("/predict")
def predict(data: Data) -> dict:
    """_summary_

    Args:
        data (Data): _description_

    Returns:
        _type_: _description_
    """

    # Extract data in correct order
    df = pd.DataFrame(jsonable_encoder(data), index=[0])

    # Create and return prediction
    prediction = model.predict(df)

    return {"prediction": prediction[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
