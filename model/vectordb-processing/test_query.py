import os
import time
import chromadb
from sentence_transformers import SentenceTransformer

# --- CẤU HÌNH (Phải khớp với cấu hình trong ingest_pdf.py) ---
DB_PATH = "chroma_db"
COLLECTION_NAME = "pet_medical_docs"
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

def init_system():
    """Khởi tạo kết nối DB và tải model Embedding."""
    print("=" * 60)
    print("🚀 ĐANG KHỞI ĐỘNG HỆ THỐNG TÌM KIẾM VECTOR (TEST MODE)")
    print("=" * 60)
    
    # 1. Kiểm tra xem DB đã được tạo chưa
    if not os.path.exists(DB_PATH):
        print(f"[LỖI] Không tìm thấy thư mục Database '{DB_PATH}'.")
        print("      Vui lòng chạy script 'ingest_pdf.py' trước để tạo dữ liệu!")
        exit(1)
        
    start_time = time.time()
    
    # 2. Kết nối Local ChromaDB
    print("[1/2] Đang kết nối tới ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        doc_count = collection.count()
        print(f"      -> Kết nối thành công! Collection '{COLLECTION_NAME}' đang có {doc_count} chunks.")
    except Exception as e:
        print(f"[LỖI] Collection '{COLLECTION_NAME}' không tồn tại trong DB.")
        exit(1)
        
    # 3. Nạp Model Embedding
    print(f"[2/2] Đang tải mô hình ngôn ngữ '{EMBEDDING_MODEL}' (CPU)...")
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING) # Ẩn bớt log rác
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    
    # 4. Warm-up model (Khởi động nóng để tránh Cold Start cho lần search đầu tiên)
    print("[3/3] Đang Warm-up mô hình...")
    embedder.encode(["chó mèo"], show_progress_bar=False)
    
    end_time = time.time()
    print(f"\n✅ Hệ thống sẵn sàng sau {end_time - start_time:.2f} giây!")
    print("-" * 60)
    
    return collection, embedder

def search(query, collection, embedder, top_k=3):
    """Hàm thực thi tìm kiếm ngữ nghĩa."""
    start_time = time.time()
    
    # 1. Chuyển đổi câu hỏi thành Vector
    query_vector = embedder.encode([query], show_progress_bar=False).tolist()
    
    # 2. Truy vấn vào ChromaDB
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )
    
    search_time = time.time() - start_time
    
    # 3. Trích xuất kết quả
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    print(f"\n[⏱ Thời gian phản hồi: {search_time:.3f}s] Kết quả tốt nhất cho: '{query}'\n")
    
    if not documents:
        print("❌ Không tìm thấy thông tin phù hợp trong DB!")
        return
        
    for i in range(len(documents)):
        # ChromaDB dùng Cosine Distance (Càng nhỏ càng tốt, 0 là giống hệt).
        # Ta chuyển ngược lại thành Tỷ lệ tự tin (Confidence %) cho dễ hiểu.
        similarity_score = (1 - distances[i]) * 100 
        
        page_num = metadatas[i].get('page', 'N/A')
        
        print(f"🔹 TOP {i+1} | Tự tin: {similarity_score:.1f}% | Nguồn: Trang {page_num}")
        print(f"Nội dung:\n{documents[i]}")
        print("-" * 60)

def main():
    # Khởi tạo DB và Model (Chỉ load 1 lần để tiết kiệm thời gian)
    collection, embedder = init_system()
    
    print("\n💡 GỢI Ý: Hãy thử gõ các câu hỏi như: ")
    print("   - Chó bị dại có triệu chứng gì?")
    print("   - Làm sao để chữa tiêu chảy cho mèo?")
    print("   (Gõ 'exit' hoặc 'quit' để thoát chương trình)\n")
    
    # Vòng lặp chat tương tác
    while True:
        try:
            user_query = input("\n📝 Nhập câu hỏi y khoa: ").strip()
            
            if user_query.lower() in ['exit', 'quit', 'thoát']:
                print("Tạm biệt!")
                break
                
            if not user_query:
                continue
                
            search(user_query, collection, embedder, top_k=3)
            
        except KeyboardInterrupt: # Bắt sự kiện Ctrl+C
            print("\nĐã hủy. Tạm biệt!")
            break

if __name__ == "__main__":
    main()