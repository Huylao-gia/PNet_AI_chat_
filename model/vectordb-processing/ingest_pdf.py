import fitz  # PyMuPDF
import re
import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- CẤU HÌNH ---
PDF_PATH = "data/knowledge-base.pdf"
DB_PATH = "chroma_db"
COLLECTION_NAME = "pet_medical_docs"
# Model này siêu nhẹ (~400MB), hỗ trợ tiếng Việt cực tốt, ngốn cực ít RAM
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert" 

def clean_medical_text(text):
    """
    Chiến lược làm sạch sâu (Deep Clean):
    Giáo trình PDF thường bị lỗi ngắt dòng giữa câu, số trang xen ngang.
    """
    if not text:
        return ""
        
    # 1. Xóa các ký tự điều khiển không in được (trừ \n)
    text = re.sub(r'[\x00-\x09\x0b-\x1f\x7f-\x9f]', '', text)
    
    # 2. Xóa các mẫu giống số trang đứng đơn độc (VD: " - 12 - " hoặc chỉ số "12" ở cuối dòng)
    text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 3. SỬA LỖI NGẮT DÒNG GIỮA CÂU TỪ PDF (QUAN TRỌNG NHẤT)
    # Nếu một dòng kết thúc mà không có dấu câu (.!?:;) và dòng tiếp theo viết thường,
    # đó chắc chắn là một câu bị ngắt xuống dòng. Ta nối chúng lại bằng khoảng trắng.
    # Pattern: Ký tự chữ/số -> Xuống dòng -> Ký tự viết thường tiếng Việt
    vietnamese_lower = "a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    text = re.sub(rf'(?<=[^\.\!\?\:\;])\n(?=[{vietnamese_lower}])', ' ', text)
    
    # 4. Chuẩn hóa khoảng trắng & dòng trống
    text = re.sub(r'\n{3,}', '\n\n', text) # Giảm nhiều dòng trống liên tiếp xuống tối đa 2 dòng
    text = re.sub(r'[ \t]+', ' ', text)    # Gom nhiều space thành 1 space
    
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """Trích xuất và làm sạch text từ PDF, trả về danh sách các trang."""
    print(f"[*] Đang nạp và phân tích cấu trúc file PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages_text = []
    
    for page_num in tqdm(range(len(doc)), desc="Parsing & Cleaning Pages"):
        page = doc.load_page(page_num)
        
        # get_text("blocks") giúp giữ thứ tự đọc tự nhiên hơn (từ trên xuống, trái sang phải)
        blocks = page.get_text("blocks")
        # Mỗi block là (x0, y0, x1, y1, text, block_no, block_type)
        page_content = "\n".join([block[4] for block in blocks if block[6] == 0]) # Chỉ lấy text (type 0)
        
        cleaned_text = clean_medical_text(page_content)
        
        # Bỏ qua các trang trống hoặc có quá ít nội dung (< 50 ký tự)
        if len(cleaned_text) > 50: 
            pages_text.append({
                "page": page_num + 1,
                "content": cleaned_text
            })
    return pages_text

def process_and_ingest():
    # 1. Trích xuất & Làm sạch
    if not os.path.exists(PDF_PATH):
        print(f"[LỖI] Không tìm thấy file {PDF_PATH}. Vui lòng kiểm tra lại cấu trúc thư mục.")
        return
        
    pages_data = extract_text_from_pdf(PDF_PATH)
    
    # 2. Chiến lược Chunking Tối ưu
    # - chunk_size: 400 ký tự (khoảng 50 - 100 từ). Là kích thước vàng để Embedding Model
    #   "ôm trọn" được 1 ý nghĩa y khoa duy nhất mà không bị loãng.
    # - chunk_overlap: 150 ký tự. Đảm bảo nếu một định nghĩa bị cắt làm đôi, 
    #   cả 2 chunk đều chứa phần liên kết.
    print("\n[*] Đang tiến hành Semantic Chunking...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        # Ưu tiên ngắt ở đoạn văn (\n\n) -> câu (\n, .) -> phẩy (,) -> từ ( )
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
    )
    
    documents = []
    metadatas = []
    ids = []
    
    chunk_id_counter = 1
    for page in pages_data:
        chunks = text_splitter.split_text(page["content"])
        for chunk in chunks:
            # Bỏ qua các chunk quá ngắn (như mục lục, tiêu đề đơn độc)
            if len(chunk) < 50:
                continue
                
            documents.append(chunk)
            # Thêm metadata để sau này có thể filter theo trang hoặc độ dài
            metadatas.append({
                "source": "knowledge_base",
                "page": page["page"],
                "length": len(chunk)
            })
            ids.append(f"doc_{chunk_id_counter}")
            chunk_id_counter += 1
            
    print(f"[+] Đã chia nhỏ giáo trình thành {len(documents)} chunks chất lượng cao.")

    # 3. Khởi tạo Embedding Model & VectorDB
    # Ép chạy trên CPU ("cpu") để giải phóng RAM tối đa. Model sbert nhỏ nên chạy CPU vẫn cực kỳ nhanh.
    print(f"\n[*] Đang khởi động Neural Embedding Model: {EMBEDDING_MODEL} (CPU mode)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    
    print("[*] Đang kết nối Local ChromaDB...")
    # Tạo thư mục nếu chưa có
    os.makedirs(DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Cài đặt ghi đè: Xóa collection cũ nếu chạy lại script
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("    -> Đã dọn dẹp Database cũ.")
    except Exception:
        pass
        
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # Cosine là chuẩn vàng đo tương đồng văn bản
    )
    
    # 4. Ingest data vào VectorDB (Sử dụng Batching để quản lý RAM)
    # Batch size 64 là lý tưởng cho CPU: cân bằng giữa tốc độ và dung lượng RAM tiêu thụ
    batch_size = 64
    print("\n[*] BẮT ĐẦU MÃ HÓA VÀ LƯU TRỮ VÀO DATABASE...")
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Vectorizing & Ingesting"):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # show_progress_bar=False vì đã có tqdm tổng bao bên ngoài
        batch_embeddings = embedder.encode(batch_docs, show_progress_bar=False).tolist()
        
        # Batch insert vào ChromaDB (Rất nhanh do lưu trên ổ cứng local)
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metas,
            ids=batch_ids
        )
        
    print(f"\n[🚀 THÀNH CÔNG] Toàn bộ dữ liệu thú y đã được Vector hóa an toàn tại: {DB_PATH}/")
    print(f"             Tổng số Documents (Chunks) đang lưu trữ: {len(documents)}")

if __name__ == "__main__":
    process_and_ingest()
