# filepath: backend/services/memory.py
from typing import List, Dict

# Giả lập In-memory DB. Trong thực tế (Production quy mô lớn), bạn nên thay bằng Redis.
SESSIONS: Dict[str, List[Dict[str, str]]] = {}

def get_chat_history(session_id: str, max_messages: int = 6) -> List[Dict[str, str]]:
    """Lấy lịch sử chat. Mặc định lấy 6 tin gần nhất (3 lượt hỏi - đáp)."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []
    return SESSIONS[session_id][-max_messages:]

def add_message(session_id: str, role: str, content: str):
    """Thêm tin nhắn mới vào session. Role có thể là 'user' hoặc 'assistant'."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []
        
    SESSIONS[session_id].append({"role": role, "content": content})
    
    # Garbage Collection: Không giữ quá 20 tin nhắn mỗi user để tiết kiệm RAM server
    if len(SESSIONS[session_id]) > 20:
        SESSIONS[session_id] = SESSIONS[session_id][-20:]
