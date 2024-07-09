from pydantic import BaseModel


class SummaryModel(BaseModel):
    text: str


class QAModel(BaseModel):
    query: str
    context: str
