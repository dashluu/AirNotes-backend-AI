from transformers import pipeline
from llama_index.core import Document, get_response_synthesizer
from llama_index.core import DocumentSummaryIndex, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from bs4 import BeautifulSoup


class NLPEngine:
    def __init__(self, model="llama3", embedding="nomic-embed-text") -> None:
        # Model setup with Llama 3
        Settings.llm = Ollama(model=model, request_timeout=60.0)
        # Each chunk is a node
        # Chunk size by default is 1024 and chunk overlap by default is 20
        Settings.node_parser = SentenceSplitter()
        # Use Nomic embedding
        Settings.embed_model = OllamaEmbedding(model_name=embedding)
        self.__response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT, use_async=True
        )

    def __remove_html(self, html_text):
        soup = BeautifulSoup(html_text, features="html.parser")
        return soup.get_text()

    async def summarize(self, html_text):
        text = self.__remove_html(html_text)
        docs = [Document(text=text)]
        index = DocumentSummaryIndex.from_documents(
            docs, response_synthesizer=self.__response_synthesizer, show_progress=True
        )
        summary = index.get_document_summary(docs[0].doc_id)
        return summary

    def answer_question(self, query, html_context):
        context = self.__remove_html(html_context)
        docs = [Document(text=context)]
        index = VectorStoreIndex.from_documents(
            docs, response_synthesizer=self.__response_synthesizer, show_progress=True
        )
        # similarity_top_k is the number of nodes with top similarity, default is 1
        # similarity_cutoff is the least similarity required for a node to be chosen
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
        response = query_engine.query(query)
        # Generator for the response
        return response.response_gen
