from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="ID phiên chat của User, dùng để lưu lịch sử")
    message: str = Field(..., description="Câu hỏi y khoa từ người dùng")
    top_k: int = Field(default=3, description="Số lượng kết quả lấy từ VectorDB")

class ContextDocument(BaseModel):
    page: str
    content: str
    confidence_score: float
