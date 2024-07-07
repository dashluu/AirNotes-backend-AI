from transformers import pipeline
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3", request_timeout=60.0)

qa_model = "deepset/roberta-base-squad2"
qa = pipeline("question-answering", model=qa_model, tokenizer=qa_model, device="mps")


def summarize(text):
    result = llm.complete(f"Summarize the following text:\n{text}")
    return result.text


def answer_question(question, context):
    qa_input = {"question": question, "context": context}
    result = qa(qa_input)
    return result
