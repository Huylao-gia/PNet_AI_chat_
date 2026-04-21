
# HƯỚNG DẪN CHI TIẾT: TỪ PDF GIÁO TRÌNH ĐẾN VECTOR DATABASE

## 1. Cấu trúc thư mục làm việc

Bạn hãy tổ chức thư mục `vectordb-processing/` như sau:

```
vectordb-processing/
├── data/
│   └── giao_trinh_thu_y.pdf       # File PDF giáo trình 70 trang của bạn
├── chroma_db/                     # (Thư mục này sẽ tự động sinh ra sau khi chạy code)
├── requirements.txt               # Danh sách thư viện cần thiết
├── ingest_pdf.py                  # Code: Đọc PDF, cắt đoạn, tạo embedding và lưu DB
└── test_query.py                  # Code: Kiểm thử truy vấn xem DB hoạt động đúng không

```

## 2. Cài đặt môi trường (requirements.txt)

Tạo file `requirements.txt` và thêm các dòng sau. Sau đó chạy lệnh `pip install -r requirements.txt`.

```
PyMuPDF==1.23.22           # Thư viện đọc file PDF (tên import là fitz)
langchain-text-splitters==0.0.1  # Thư viện hỗ trợ cắt text thông minh
sentence-transformers==2.5.1     # Chạy model Embedding
chromadb==0.4.24           # Vector Database
tqdm==4.66.2               # Hiển thị thanh tiến trình

```

## 3. Script xử lý chính (ingest_pdf.py)

Script này thực hiện quy trình chuẩn: **Parse PDF -> Clean Text -> Chunking -> Embedding -> Lưu ChromaDB**. Bạn tạo file `ingest_pdf.py` với nội dung sau:

```
import fitz  # PyMuPDF
import re
import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- CẤU HÌNH ---
PDF_PATH = "data/giao_trinh_thu_y.pdf"
DB_PATH = "chroma_db"
COLLECTION_NAME = "pet_medical_docs"
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

def clean_text(text):
    """Làm sạch text cơ bản: Xóa khoảng trắng thừa, nối dòng bị ngắt."""
    # Thay thế nhiều khoảng trắng hoặc dòng mới bằng 1 khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    # Loại bỏ các ký tự đặc biệt không cần thiết (tùy chọn)
    text = text.strip()
    return text

def extract_text_from_pdf(pdf_path):
    """Trích xuất text từ PDF, trả về danh sách các trang."""
    print(f"Đang đọc file PDF: {pdf_path}...")
    doc = fitz.open(pdf_path)
    pages_text = []
    
    for page_num in tqdm(range(len(doc)), desc="Parsing Pages"):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        cleaned_text = clean_text(text)
        if len(cleaned_text) > 50: # Bỏ qua các trang trống hoặc quá ít chữ
            pages_text.append({
                "page": page_num + 1,
                "content": cleaned_text
            })
    return pages_text

def process_and_ingest():
    # 1. Trích xuất text
    if not os.path.exists(PDF_PATH):
        print(f"Lỗi: Không tìm thấy file {PDF_PATH}")
        return
        
    pages_data = extract_text_from_pdf(PDF_PATH)
    
    # 2. Chunking (Cắt đoạn text)
    # Cắt khoảng 1000 ký tự mỗi chunk (~200-250 từ), overlap 200 ký tự để giữ ngữ cảnh liền mạch
    print("Đang cắt văn bản (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    documents = []
    metadatas = []
    ids = []
    
    chunk_id_counter = 1
    for page in pages_data:
        chunks = text_splitter.split_text(page["content"])
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append({"source": "giao_trinh_thu_y", "page": page["page"]})
            ids.append(f"chunk_{chunk_id_counter}")
            chunk_id_counter += 1
            
    print(f"Đã tạo ra {len(documents)} chunks từ file PDF.")

    # 3. Khởi tạo Embedding Model & VectorDB
    print(f"Đang tải Embedding Model: {EMBEDDING_MODEL} (CPU)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Đang khởi tạo ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Xóa collection cũ nếu tồn tại để chạy lại từ đầu
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except:
        pass
        
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # Tối ưu đo lường độ tương đồng cho vector
    )
    
    # 4. Tạo Vector và chèn vào DB theo từng batch (để tránh tràn RAM nếu data lớn)
    batch_size = 100
    print("Đang tạo Embeddings và lưu vào ChromaDB...")
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Ingesting to DB"):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # Tạo vectors
        batch_embeddings = embedder.encode(batch_docs).tolist()
        
        # Thêm vào DB
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metas,
            ids=batch_ids
        )
        
    print(f"\n[THÀNH CÔNG] Dữ liệu đã được lưu trữ an toàn tại thư mục: {DB_PATH}/")

if __name__ == "__main__":
    process_and_ingest()

```

## 4. Kiểm thử Dữ liệu (test_query.py)

Để chắc chắn vectorDB hoạt động hiệu quả, hãy tạo file `test_query.py` để tìm kiếm thử một câu hỏi liên quan đến bệnh lý.

```
from sentence_transformers import SentenceTransformer
import chromadb

DB_PATH = "chroma_db"
COLLECTION_NAME = "pet_medical_docs"
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

def test_search(query):
    # Load lại DB và Model
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"Câu hỏi: {query}\n")
    
    # Mã hóa câu hỏi thành vector
    query_vector = embedder.encode([query]).tolist()
    
    # Tìm kiếm top 3 đoạn văn bản liên quan nhất
    results = collection.query(
        query_embeddings=query_vector,
        n_results=3
    )
    
    # Hiển thị kết quả
    for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
        print(f"--- KẾT QUẢ TOP {i+1} (Độ tương đồng Cosine: {1 - dist:.4f}) ---")
        print(f"Trang PDF: {meta['page']}")
        print(f"Nội dung: {doc}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    # Thay đổi câu hỏi này bằng một câu hỏi có thật trong giáo trình của bạn
    test_search("Chó bị nôn mửa, tiêu chảy ra máu là dấu hiệu của bệnh gì?")

```

## 5. Quy trình thực thi

1.  Chuẩn bị file PDF bỏ vào `vectordb-processing/data/` (đổi tên trong code thành `giao_trinh_thu_y.pdf` hoặc sửa biến `PDF_PATH` trong code).
    
2.  Chạy lệnh: `python ingest_pdf.py` _(Tiến trình sẽ mất khoảng 1-3 phút tùy thuộc vào tốc độ CPU của máy tính)._
    
3.  Chạy lệnh: `python test_query.py` để kiểm tra chất lượng cắt ghép và độ chính xác của tìm kiếm.
    

## Tích hợp vào hệ thống (Bước tiếp theo)

Sau khi script trên chạy xong, bạn sẽ thấy thư mục `chroma_db` (bên trong chứa các file SQLite và Parquet). Khi build hệ thống backend, bạn **không cần chạy lại script này nữa**. Bạn chỉ việc Copy/Mount thư mục `chroma_db` này vào container của FastAPI, và dùng đúng các hàm trong `test_query.py` để gọi data đưa vào prompt là xong!
