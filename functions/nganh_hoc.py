import os

from rag import RagMini
from retriever import Retriever
from core.models import get_chat_completion

data_path = os.path.join(
    os.getcwd(), "rag_n_function_calling", "data", "txt_data", "Ngành và chương trình học"
)

class NganhVaChuyenNganh(RagMini):
    def __init__(self):
        self.retriever = Retriever(data_path, "multi_folders")
        self.description = "Cung cấp thông tin chung và riêng về ngành, chuyên ngành và chương trình học mà bạn cần biết."

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

nganh_va_chuyen_nganh_rag = NganhVaChuyenNganh()

        


