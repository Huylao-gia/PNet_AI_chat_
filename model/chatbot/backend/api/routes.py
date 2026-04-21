# filepath: backend/api/routes.py
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from schemas.models import ChatRequest
from services import vector_store, memory, rag_engine, llm_client

router = APIRouter()

@router.post("/api/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    session_id = body.session_id
    user_msg = body.message
    
    # 1. Lưu ngay câu hỏi của người dùng vào Session Memory
    memory.add_message(session_id, "user", user_msg)
    
    # 2. Truy xuất tài liệu từ VectorDB siêu tốc
    contexts = await vector_store.search_context(user_msg, request, top_k=body.top_k)
    
    # 3. Lấy lịch sử chat (đã bao gồm câu hỏi vừa lưu ở bước 1)
    history = memory.get_chat_history(session_id, max_messages=6)
    
    # 4. Gói ghém tất cả thành Prompt chuẩn mực
    messages = rag_engine.build_prompt_messages(contexts, history)
    
    # 5. Hàm Generator phát sự kiện SSE
    async def event_generator():
        full_ai_response = ""
        try:
            # Lặp qua từng Token được vLLM nhả ra
            async for token in llm_client.stream_generate(messages):
                full_ai_response += token
                # Đóng gói Token thành chuỗi JSON an toàn cho giao thức SSE
                payload = json.dumps({"content": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                
        finally:
            # 6. KHI HOÀN THÀNH (hoặc khi user tắt trình duyệt giữa chừng):
            # Lưu lại toàn bộ câu trả lời của AI vào Memory để dùng cho lượt hỏi sau
            if full_ai_response:
                memory.add_message(session_id, "assistant", full_ai_response)
            
            # Báo hiệu cho Website biết đã stream xong
            yield "data: [DONE]\n\n"

    # Trả về Response chuẩn Streaming
    return StreamingResponse(event_generator(), media_type="text/event-stream")
