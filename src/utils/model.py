import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_model() -> list:

    features_to_transform = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "AvgBedsPerRoom",
    ]

    transformer = Pipeline(steps=[("standard_scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("std", transformer, features_to_transform),
        ]
    )

    linear_regressor = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    )

    random_forest = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor())]
    )

    return [linear_regressor, random_forest]


def select_best_model(
    models: list, X: pd.DataFrame, y: pd.DataFrame, cv: int = 3, scoring: str = "r2"
) -> Pipeline:
    """Seleciona o modelo que obtiver o melhor score

    Args:
        models (list): lista contendo os modelos
        X (pd.DataFrame): Matriz de features para treino
        y (pd.DataFrame): Variável resposta
        cv (int, optional): Número de validações cruzadas. Defaults to 3.
        scoring (str, optional): Forma de scorar o modelo. Defaults to "r2".

    Returns:
        sklearn.pipeline.Pipeline: Pipeline do modelo
    """
    scores = []
    for model in models:
        scores.append(cross_val_score(model, X, y, cv=cv, scoring=scoring).mean())

    best_model_position = np.array(scores).argmax()
    return models[best_model_position]


def get_best_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    models = create_model()
    best_model = select_best_model(models, X_train, y_train)
    return best_model.fit(X_train, y_train)


def save_model(model: Pipeline, path: str, compress: str = "gzip"):
    return joblib.dump(model, path, compress=compress)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def evaluation(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)
