from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from sentence_transformers import SentenceTransformer

from core.config import settings
from api.routes import router as chat_router

# Quản lý Lifespan: Load model 1 lần duy nhất khi bật server
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print(f"🚀 KHỞI ĐỘNG HỆ THỐNG: {settings.PROJECT_NAME}")
    print("=" * 50)
    
    # 1. Khởi tạo Vector DB
    print("[1/2] Đang kết nối ChromaDB...")
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    app.state.collection = client.get_or_create_collection(name=settings.COLLECTION_NAME)
    
    # 2. Khởi tạo Embedding Model
    print(f"[2/2] Đang nạp SBERT: {settings.EMBEDDING_MODEL} (CPU)...")
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    app.state.embedder = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
    
    # Warm-up (Tránh Cold start)
    app.state.embedder.encode(["warm up"], show_progress_bar=False)
    
    print("✅ Hệ thống đã sẵn sàng xử lý Request!")
    print("=" * 50)
    
    yield # Máy chủ chạy tại đây
    
    print("🛑 Đang tắt hệ thống, giải phóng bộ nhớ...")
    app.state.collection = None
    app.state.embedder = None

# Khởi tạo App
app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Cấu hình CORS (Cho phép Website gọi API không bị lỗi Cross-Origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Trên production nên thay bằng URL của website
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạm thời tạo router trống ở routes.py để tránh lỗi khi import
# Bổ sung dòng sau vào backend/api/routes.py:
# from fastapi import APIRouter
# router = APIRouter()

# Gắn các API endpoint vào app chính
app.include_router(chat_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    # Mặc định chạy trên cổng 8080
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
    
    
# RUNNING and CHECKING GUIDE
# cd backend
# python main.py
# *(Bạn sẽ thấy Log báo nạp DB và Warm-up thành công)*

# **Bước 2:** Bật thêm một cửa sổ Terminal mới (Giả lập Website gọi tới) và chạy lệnh `curl` này:

# ```bash
# curl -N -X POST http://localhost:8080/api/chat \
#      -H "Content-Type: application/json" \
#      -d '{"session_id": "test_001", "message": "Chó nhà tôi bị nôn mửa, phải làm sao?"}'

# **Kết quả kỳ vọng:**
# Nếu phía dưới con vLLM của bạn đang chạy (ở cổng 8000), bạn sẽ thấy trên màn hình Terminal của cURL in ra từng cụm `data: {"content": "..."}` giống hệt như cách ChatGPT hiện chữ lên trình duyệt!

# Nếu bạn chưa bật vLLM, API sẽ báo lỗi Connection Refused (do chưa có AI). Vậy bạn đã cài đặt chạy thử vLLM Engine với file `.gguf` bao giờ chưa? Nếu chưa, Phase tiếp theo tôi sẽ hướng dẫn bạn viết kịch bản **Docker Compose (Phase 4)** để ép vLLM chạy ngầm nhé.
