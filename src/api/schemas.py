# pylint: disable=no-name-in-module
# pylint: disable=no-self-argument
from pydantic import BaseModel


class Data(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    satisfaction_level: float
    last_evaluation: float
    number_project: float
    average_montly_hours: float
    time_spend_company: float
    Work_accident: float
    promotion_last_5years: float
    sales: str
    salary: str
