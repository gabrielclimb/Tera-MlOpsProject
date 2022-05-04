
import pickle

import lightgbm
import numpy as np
import uvicorn
from fastapi import FastAPI

from schemas import Data

app = FastAPI()

# Initialize files
clf = pickle.load(open("model.pickle", "rb"))
enc = pickle.load(open("encoder.pickle", "rb"))
features = pickle.load(open("features.pickle", "rb"))


@app.post("/predict")
async def predict(data: Data):

    # Extract data in correct order
    data_dict = data.dict()
    to_predict = [data_dict[feature] for feature in features]

    # Apply one-hot encoding
    encoded_features = list(enc.transform(np.array(to_predict[-2:]).reshape(1, -1))[0])
    to_predict = np.array(to_predict[:-2] + encoded_features)

    # Create and return prediction
    prediction = clf.predict(to_predict.reshape(1, -1))

    return {"prediction": int(prediction[0])}
