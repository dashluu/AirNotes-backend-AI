from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup

from tasks.nlp import summarize, answer_question
from models.nlp_models import SummaryModel, QAModel

app = FastAPI()

# Define origins for CORS
origins = [
    "http://localhost",
    "http://localhost:5173"
]

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/summarize", response_model=str)
def summarize_handler(summary_model: SummaryModel):
    soup = BeautifulSoup(summary_model.text, features="html.parser")
    text_to_summarize = soup.get_text()
    return summarize(text_to_summarize)


@app.post("/qa", response_model=str)
def qa_handler(qa_model: QAModel):
    question = qa_model.question
    soup = BeautifulSoup(qa_model.context, features="html.parser")
    context = soup.get_text()
    result = answer_question(question, context)
    return result["answer"]
