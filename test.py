from functions import (
    chi_tieu_tuyen_sinh,
    nganh_hoc,
    phuong_thuc_xet_tuyen,
    quy_doi_chung_chi,
    diem_trung_tuyen,
    thong_tin_truong,
    hoc_bong,
    ky_tuc_xa,
    le_phi
)
from core.models import get_chat_completion

functions = {
    "PhuongThucXetTuyen": phuong_thuc_xet_tuyen.phuong_thuc_xet_tuyen_rag,
    "ChiTieuTuyenSinh": chi_tieu_tuyen_sinh.chi_tieu_tuyen_sinh_rag,
    "QuyDoiChungChi": quy_doi_chung_chi.quy_doi_chung_chi_rag,
    "NganhVaChuyenNganh": nganh_hoc.nganh_va_chuyen_nganh_rag,
    "DiemTrungTuyen": diem_trung_tuyen.diem_trung_tuyen_rag,
    "ThongTinTruong": thong_tin_truong.thong_tin_truong_rag,
    "HocBong": hoc_bong.hoc_bong_rag,
    "KyTucXa": ky_tuc_xa.ky_tuc_xa_rag,
    "LePhi": le_phi.le_phi_rag
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

print(functions_calling(question = "Cho tôi toàn bộ thông tin về trường"))
    


    
