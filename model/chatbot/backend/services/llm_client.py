# filepath: backend/services/llm_client.py
from openai import AsyncOpenAI
from core.config import settings

# Khởi tạo Async Client. Trỏ base_url tới vLLM (Ví dụ: http://localhost:8000/v1)
client = AsyncOpenAI(
    base_url=settings.VLLM_API_URL,
    api_key="EMPTY_KEY" # vLLM cục bộ không cần API Key
)

async def stream_generate(messages: list):
    """Gọi vLLM và trả về từng chữ (token) một ngay lập tức."""
    response = await client.chat.completions.create(
        model="pet-chat-model", # Tên model phải khớp với lệnh khởi chạy vLLM
        messages=messages,
        stream=True,         # Bật Streaming
        temperature=0.1,     # Nhiệt độ thấp (0.1) giúp AI không bị "ảo giác", trả lời chuẩn theo y khoa
        max_tokens=1024,
        presence_penalty=0.2 # Khuyến khích AI dùng từ đa dạng một chút
    )
    
    async for chunk in response:
        # Lấy token vừa được sinh ra (nếu có)
        if chunk.choices and chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
