from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid

from engines.nlp_engine import NLPEngine
from engines.vision_engine import VisionEngine
from models.nlp_models import SummaryModel, QAModel, SearchModel
from models.vision_models import TextToImgModel
from config import FIREBASE_DB_LOCAL
import json
import nest_asyncio

nest_asyncio.apply()

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
    if summary_model.text == "document":
        with open(
            f"{FIREBASE_DB_LOCAL}/{summary_model.user_id}/{summary_model.doc_id}.json",
            "r+",
        ) as f:
            doc = json.load(f)
        text = doc["content"]
    else:
        text = summary_model.text
    # summary = nlp_engine.summarize(text)
    # return summary
    ans_gen = await nlp_engine.summarize(text)
    return StreamingResponse(ans_gen)


@app.post("/qa", response_model=str)
async def qa_handler(qa_model: QAModel):
    if qa_model.context == "document":
        with open(
            f"{FIREBASE_DB_LOCAL}/{qa_model.user_id}/{qa_model.doc_id}.json", "r+"
        ) as f:
            doc = json.load(f)
        context = doc["content"]
    else:
        context = qa_model.context
    ans_gen = await nlp_engine.answer_question(qa_model.query, context)
    return StreamingResponse(ans_gen)


@app.post("/text-to-img")
def text_to_img_handler(text_to_img_model: TextToImgModel):
    text = text_to_img_model.text
    img_id = str(uuid.uuid4())
    img_name = f"images/{img_id}.jpg"
    vision_engine.text_to_img(text, img_name)
    return FileResponse(img_name)


@app.post("/search", response_model=list[dict])
async def search_list_handler(search_model: SearchModel):
    user_id = search_model.user_id
    query = search_model.query
    results = await nlp_engine.search(user_id, query)
    return results
