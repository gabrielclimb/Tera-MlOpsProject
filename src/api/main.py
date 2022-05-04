import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from .schemas import Data

# Initialize files
model = joblib.load("src/model/regressor.gzip")


app = FastAPI()


@app.post("/")
def hello() -> str:
    """just a standard route

    Returns:
        str: hello world
    """
    return "Hello World, go to /docs"


@app.post("/predict")
def predict(data: Data):
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


# {
#     "MedInc": 1.6812,
#     "HouseAge": 25.0,
#     "AveRooms": 4.192200557103064,
#     "AveBedrms": 1.0222841225626742,
#     "Population": 1392.0,
#     "AveOccup": 3.8774373259052926,
#     "Latitude": 36.06,
#     "Longitude": -119.01,
#     "AvgBedsPerRoom": 0.24385382059800667,
# }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
