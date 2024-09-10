from pydantic import BaseModel


class SummaryModel(BaseModel):
    user_id: str
    doc_id: str
    text: str


class QAModel(BaseModel):
    user_id: str
    doc_id: str
    query: str
    context: str


class SearchModel(BaseModel):
    user_id: str
    query: str
