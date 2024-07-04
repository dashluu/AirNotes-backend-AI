from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
import uuid

from tasks.nlp import summarize, answer_question
from tasks.vision import text_to_img
from models.nlp_models import SummaryModel, QAModel
from models.vision_models import TextToImgModel

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
def summarization_handler(summary_model: SummaryModel):
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


@app.post("/text-to-img")
def text_to_img_handler(text_to_img_model: TextToImgModel):
    text = text_to_img_model.text
    img_id = str(uuid.uuid4())
    img_name = f"images/{img_id}.jpg"
    text_to_img(text, img_name)
    return FileResponse(img_name)
