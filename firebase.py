"""Firebase Realtime Database Loader."""

from typing import List, Optional

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from bs4 import BeautifulSoup
from config import (
    FIREBASE_DB_URL,
    FIREBASE_SERVICE_KEY_PATH,
    FIREBASE_NOTE_COLLECTION,
)


class FirebaseFirestoreReader(BaseReader):
    """Firebase Realtime Database reader.

    Retrieves data from Firebase Realtime Database and converts it into the Document used by LlamaIndex.

    Args:
        database_url (str): Firebase Realtime Database URL.
        service_account_key_path (Optional[str]): Path to the service account key file.

    """

    def __init__(
        self,
        database_url: str,
        service_account_key_path: Optional[str] = None,
    ) -> None:
        """Initialize with parameters."""
        cred = credentials.Certificate(service_account_key_path)
        self.app = firebase_admin.initialize_app(
            cred, options={"databaseURL": database_url}
        )

    def load_data(self, collection: str) -> List[Document]:
        """Load data from Firebase Realtime Database and convert it into documents.

        Args:
            collection (str): Name of the collection to be read from the Firestore.

        """
        client = firestore.client(self.app)
        doc_list = client.collection(collection).get()
        for doc in doc_list:
            with open(f"db/{doc.id}.txt", "w") as f:
                doc_dict = doc.to_dict()
                html_content = doc_dict["content"]
                soup = BeautifulSoup(html_content, features="html.parser")
                content = soup.get_text()
                f.write(content)


def init_db():
    reader = FirebaseFirestoreReader(FIREBASE_DB_URL, FIREBASE_SERVICE_KEY_PATH)
    reader.load_data(FIREBASE_NOTE_COLLECTION)
    return reader


firestore_reader = init_db()
