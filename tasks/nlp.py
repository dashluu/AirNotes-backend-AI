from transformers import pipeline
from llama_index.core import Document, get_response_synthesizer
from llama_index.core import DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

Settings.llm = Ollama(model="llama3", request_timeout=60.0)
Settings.node_parser = SentenceSplitter()
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

qa_model = "deepset/roberta-base-squad2"
qa = pipeline("question-answering", model=qa_model, tokenizer=qa_model, device="mps")


def summarize(text):
    docs = [Document(text=text)]
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize", use_async=False
    )
    summary_index = DocumentSummaryIndex.from_documents(
        docs, response_synthesizer=response_synthesizer, show_progress=True
    )
    summary = summary_index.get_document_summary(docs[0].doc_id)
    return summary


def answer_question(question, context):
    qa_input = {"question": question, "context": context}
    result = qa(qa_input)
    return result
