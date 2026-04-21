# filepath: backend/services/rag_engine.py
from typing import List
from schemas.models import ContextDocument

def build_prompt_messages(contexts: List[ContextDocument], history: List[Dict[str, str]]) -> list:
    """
    Xây dựng chuẩn tin nhắn (messages array) để gửi tới LLM.
    - System prompt (chứa Context từ DB)
    - History (chứa các câu hỏi đáp cũ + câu hỏi mới nhất của user)
    """
    # 1. Định dạng tài liệu tham khảo
    context_str = ""
    if contexts:
        for idx, ctx in enumerate(contexts):
            context_str += f"[Tài liệu {idx+1} | Trang {ctx.page} | Độ chính xác {ctx.confidence_score}%]:\n{ctx.content}\n\n"
    else:
        context_str = "Không tìm thấy tài liệu tham khảo nào trong cơ sở dữ liệu."

    # 2. Xây dựng System Prompt (Ép AI phải tuân thủ nghiêm ngặt RAG)
    system_prompt = f"""Bạn là một chuyên gia Thú y AI ảo chuyên nghiệp và tận tâm.
NHIỆM VỤ CỦA BẠN: Trả lời câu hỏi của người dùng CHỈ DỰA VÀO phần "TÀI LIỆU THAM KHẢO" dưới đây.

QUY TẮC NGHIÊM NGẶT:
1. Nếu thông tin CÓ TRONG tài liệu, hãy tổng hợp và trả lời dễ hiểu, có thể liệt kê các bước nếu cần.
2. Nếu thông tin KHÔNG CÓ TRONG tài liệu, hãy nói rõ: "Dựa trên tài liệu y khoa hiện tại, tôi không tìm thấy thông tin này...", TUYỆT ĐỐI KHÔNG tự bịa đặt kiến thức (Hallucination).
3. Sử dụng tiếng Việt chuẩn xác, giọng điệu đồng cảm với người nuôi thú cưng.

TÀI LIỆU THAM KHẢO:
{context_str}
"""
    
    # 3. Lắp ráp cấu trúc cuối cùng
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history) # Đưa toàn bộ ngữ cảnh trò chuyện vào
    
    return messages
