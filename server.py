from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid

from engines.nlp_engine import NLPEngine
from engines.vision_engine import VisionEngine
from models.nlp_models import SummaryModel, QAModel
from models.vision_models import TextToImgModel

app = FastAPI()
nlp_engine = NLPEngine()
vision_engine = VisionEngine()

# Define origins for CORS
origins = ["http://localhost", "http://localhost:5173"]

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/summarize", response_model=str)
async def summarization_handler(summary_model: SummaryModel):
    summary = await nlp_engine.summarize(summary_model.text)
    return summary


@app.post("/qa", response_model=str)
def qa_handler(qa_model: QAModel):
    ans_gen = nlp_engine.answer_question(qa_model.query, qa_model.context)
    return StreamingResponse(ans_gen)


@app.post("/text-to-img")
async def text_to_img_handler(text_to_img_model: TextToImgModel):
    text = text_to_img_model.text
    img_id = str(uuid.uuid4())
    img_name = f"images/{img_id}.jpg"
    await vision_engine.text_to_img(text, img_name)
    return FileResponse(img_name)
