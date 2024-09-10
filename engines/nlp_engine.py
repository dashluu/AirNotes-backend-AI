from transformers import pipeline
from llama_index.core import Document, get_response_synthesizer
from llama_index.core import DocumentSummaryIndex, VectorStoreIndex, StorageContext
from llama_index.core.readers.json import JSONReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from bs4 import BeautifulSoup
from firebase.firebase_reader import FirebaseFirestoreReader
from config import *
import chromadb
import json
import os
import shutil
from pathlib import Path


class NLPEngine:
    def __init__(self, model="llama3", embedding="nomic-embed-text", init_db=True):
        # Model setup with Llama 3
        Settings.llm = Ollama(model=model, request_timeout=60.0)
        # Each chunk is a node
        # Chunk size by default is 1024 and chunk overlap by default is 20
        Settings.node_parser = SentenceSplitter()
        # Use Nomic embedding
        Settings.embed_model = OllamaEmbedding(model_name=embedding)
        self.response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT, use_async=True
        )
        self.db_reader = FirebaseFirestoreReader(
            FIREBASE_DB_URL, FIREBASE_SERVICE_KEY_PATH
        )
        # Prepare firebase and vector db on initialization if necessary
        if init_db:
            self.init_db_dirs()
            self.load_search_db()
            self.init_search_indexes()

    def __remove_html(self, html_text) -> str:
        soup = BeautifulSoup(html_text, features="html.parser")
        return soup.get_text()

    async def summarize(self, html_text) -> str:
        text = self.__remove_html(html_text)
        docs = [Document(text=text)]
        # index = DocumentSummaryIndex.from_documents(
        #     docs,
        #     response_synthesizer=self.response_synthesizer,
        #     show_progress=True,
        # )
        # summary = index.get_document_summary(docs[0].doc_id)
        # return summary
        index = VectorStoreIndex(
            docs,
            response_synthesizer=self.response_synthesizer,
            use_async=True,
            show_progress=True,
        )
        query_engine = index.as_query_engine(streaming=True)
        response = await query_engine.aquery("Summarize the document")
        # Stream the response
        return response.response_gen

    async def answer_question(self, query, html_context):
        context = self.__remove_html(html_context)
        docs = [Document(text=context)]
        index = VectorStoreIndex(
            docs,
            response_synthesizer=self.response_synthesizer,
            use_async=True,
            show_progress=True,
        )
        # similarity_top_k is the number of nodes with top similarity, default is 1
        # similarity_cutoff is the least similarity required for a node to be chosen
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
        response = await query_engine.aquery(query)
        # Stream the response
        return response.response_gen

    ############################### LLM search ####################################
    def init_db_dirs(self):
        # Create a root db folder
        os.makedirs("db", exist_ok=True)
        # Overwrite Firebase db folder
        firebase_db = Path(FIREBASE_DB_LOCAL)
        if firebase_db.exists():
            shutil.rmtree(FIREBASE_DB_LOCAL)
        os.makedirs(FIREBASE_DB_LOCAL)
        # Overwite vector db folder
        vector_db = Path(VECTOR_DB_LOCAL)
        if vector_db.exists():
            shutil.rmtree(VECTOR_DB_LOCAL)
        os.makedirs(VECTOR_DB_LOCAL)

    def load_search_db(self):
        self.db_reader.load_data()

    def init_storage_context(
        self, collection_name: str
    ) -> tuple[ChromaVectorStore, StorageContext]:
        # initialize client, setting path to save data
        vector_db = chromadb.PersistentClient(path=VECTOR_DB_LOCAL)
        # create  or get collection
        collection = vector_db.get_or_create_collection(collection_name)
        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return vector_store, storage_context

    def init_search_indexes(self):
        user_dirs = os.listdir(FIREBASE_DB_LOCAL)
        for f in user_dirs:
            self.init_search_index(f)

    def init_search_index(self, user_id) -> VectorStoreIndex:
        json_reader = JSONReader()
        user_dir = os.listdir(f"{FIREBASE_DB_LOCAL}/{user_id}")
        docs = []
        for f in user_dir:
            # Append documents and provide the document ID as the citation source
            docs += json_reader.load_data(
                f"{FIREBASE_DB_LOCAL}/{user_id}/{f}",
                extra_info={"source": f.split(".")[0]},
            )
        # Use user ID as the collection name
        _, storage_context = self.init_storage_context(user_id)
        return VectorStoreIndex(
            docs,
            storage_context=storage_context,
            response_synthesizer=self.response_synthesizer,
            use_async=True,
            show_progress=True,
        )

    def load_search_index(self, user_id: str) -> VectorStoreIndex:
        vector_store, storage_context = self.init_storage_context(user_id)
        return VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            response_synthesizer=self.response_synthesizer,
            show_progress=True,
        )

    async def search(
        self, user_id: str, query: str, sim_top_k=20, top_n=20
    ) -> list[dict]:
        index = self.load_search_index(user_id)
        # create a reranker
        reranker = ColbertRerank(
            top_n=top_n,
            model="colbert-ir/colbertv2.0",
            tokenizer="colbert-ir/colbertv2.0",
            device="mps",
            keep_retrieval_score=True,
        )
        # create a query engine and query
        query_engine = index.as_query_engine(
            similarity_top_k=sim_top_k, node_postprocessors=[reranker]
        )
        response = await query_engine.aquery(f"{query}. Please cite the sources.")
        nodes = response.source_nodes
        results = []
        for i in range(len(nodes)):
            # Turn text into dictionary and append to results
            text = nodes[i].text
            text = text[: text.find(',\n"content"')]
            results.append(json.loads(f"{{{text}}}"))
        # TODO: Filter the results
        return results
