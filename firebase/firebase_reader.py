"""Firebase Realtime Database Loader."""

from typing import List, Optional

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from bs4 import BeautifulSoup
import json
import os
from config import FIREBASE_NOTE_COLLECTION, FIREBASE_DB_LOCAL


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
        self.client = firestore.client(self.app)

    def load_data(self) -> List[Document]:
        """Load data from Firebase Realtime Database and convert it into documents.

        Args:
            collection (str): Name of the collection to be read from the Firestore.

        """
        doc_list = self.client.collection(FIREBASE_NOTE_COLLECTION).get()
        for doc in doc_list:
            doc_dict = doc.to_dict()
            user_id = doc_dict["userId"]
            # Create a directory for the user if it does not exist
            os.makedirs(f"{FIREBASE_DB_LOCAL}/{user_id}", exist_ok=True)
            # Write user data to a JSON file
            with open(f"{FIREBASE_DB_LOCAL}/{user_id}/{doc.id}.json", "w+") as f:
                html_content = doc_dict["content"]
                soup = BeautifulSoup(html_content, features="html.parser")
                parsed_content = soup.get_text()
                json_data = {
                    "id": doc.id,
                    "thumbnail": doc_dict["thumbnail"],
                    "title": doc_dict["title"],
                    "lastModified": doc_dict["lastModified"].isoformat(),
                    "lastAccessed": doc_dict["lastAccessed"].isoformat(),
                    "content": parsed_content,
                }
                f.write(json.dumps(json_data))
