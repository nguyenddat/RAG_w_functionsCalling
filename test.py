from langchain.prompts import PromptTemplate

from functions.phuong_thuc_xet_tuyen import phuong_thuc_xet_tuyen_rag
from functions.chi_tieu_tuyen_sinh import chi_tieu_tuyen_sinh_rag
from functions.quy_doi_chung_chi import quy_doi_chung_chi_rag
from core.models import get_chat_completion

# print(phuong_thuc_xet_tuyen_rag.invoke(question = "Phương thức xét tuyển có sử dụng kết quả học tập THPT là gì?"))

functions = {
    "PhuongThucXetTuyen": phuong_thuc_xet_tuyen_rag,
    "ChiTieuTuyenSinh": chi_tieu_tuyen_sinh_rag,
    "QuyDoiChungChi": quy_doi_chung_chi_rag
}

functions_description = "\n".join(
    [f"{name}: {obj.description}" for name, obj in functions.items()]
)

def functions_calling(question: str) -> str:
    response = get_chat_completion(
        message = "Bạn là một trợ lý ảo thông minh. Nhiệm vụ của bạn là chọn function phù hợp nhất để xử lý câu hỏi của người dùng.",
        task = "function_calling",
        params = {
            "question": question,
            "functions_description": functions_description
        }
    )

    if response not in functions.keys():
        raise ValueError(f"Không tìm thấy function: {response}")

    else:
        print(f"Chọn function: {response}")
        return functions[response].invoke(question = question)

print(functions_calling(question = "Quy đổi chứng chỉ IELTS 6.5"))
    


    
