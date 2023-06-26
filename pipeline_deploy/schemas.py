# pylint: disable=no-name-in-module
# pylint: disable=no-self-argument
from pydantic import BaseModel


class Data(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    AvgBedsPerRoom: float