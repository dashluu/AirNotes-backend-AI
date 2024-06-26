from pydantic import BaseModel


class SummaryModel(BaseModel):
    text: str


class QAModel(BaseModel):
    question: str
    context: str
