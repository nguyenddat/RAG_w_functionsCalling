import os

from rag import RagMini
from retriever import Retriever
from core.models import get_chat_completion

data_path = os.path.join(
    os.getcwd(), "rag_n_function_calling", "data", "txt_data", "Quy đổi chứng chỉ quốc tế"
)

class QuyDoiChungChi(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path)
        self.description = "Cung cấp thông tin về cách quy đổi chứng chỉ quốc tế mà bạn cần biết."

    def get_k_relevant(self, query: str, k: int) -> str:
        docs = self.retriever.retriever.invoke(
            query, config={"k": k}
        )

        return "\n".join([doc.page_content for doc in docs])
    

    def invoke(self, question: str) -> str:
        context = self.get_k_relevant(question, 4)
        response = get_chat_completion(
            message = self.message, 
            task = "rag", 
            params = {"question": question, "context": context}
        )

        return response

quy_doi_chung_chi_rag = QuyDoiChungChi()

        


