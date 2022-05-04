from pydantic import BaseModel


class Data(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: float
    average_montly_hours: float
    time_spend_company: float
    Work_accident: float
    promotion_last_5years: float
    sales: str
    salary: str
