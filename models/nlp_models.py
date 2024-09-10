from typing import List
from pydantic import BaseModel, Field


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


class SearchOutput(BaseModel):
    response: str = Field(..., description="The answer to the question.")
    src: List[str] = Field(..., description="The document filenames of the sources.")
