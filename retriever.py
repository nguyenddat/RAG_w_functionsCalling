import os

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore

from core.models import embeddings
from helpers.data_loader import data_loader

class Retriever:
    def __init__(self, data_path):
        self.data_path = data_path
        self.build()

    def build(self):
        texts = data_loader.load_folder(self.data_path)
        index = faiss.IndexFlatL2(1024)

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        vector_store.add_documents(texts)
        vector_store.save_local(os.path.join(os.getcwd(), "rag_n_function_calling", "data", "vector_store"))
        self.retriever = VectorStoreRetriever(vectorstore=vector_store)
        return self