from typing import *
from abc import ABC, abstractmethod

class RagMini(ABC):
    retriever: Any
    description: str
    message: str = "Bạn là một trợ lý ảo của ban tuyển sinh một trường đại học. Nhiệm vụ của bạn là cung cấp và trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp."

    @abstractmethod
    def get_k_relevant(self, query: str, k: int) -> str:
        raise NotImplementedError()
    

    @abstractmethod
    def invoke(self, message: str) -> str:
        raise NotImplementedError()