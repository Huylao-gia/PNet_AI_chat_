# import all necessary libraries
import json
import os
import hashlib
import re
import unicodedata
import logging
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Thiết lập logging để theo dõi quá trình chạy pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PetDataPipeline:
    """
    Pipeline xử lý dữ liệu (Data Engineering) cho dự án Pet Care RAG Chatbot.
    Bao gồm: Hợp nhất (Consolidation), Làm sạch (Cleaning), và Khám phá (EDA).
    """

    def __init__(self, data_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Đảm bảo thư mục đầu ra tồn tại
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Đường dẫn file
        self.file_2viet = os.path.join(self.data_dir, "2viet.json")
        self.file_papddy = os.path.join(self.data_dir, "papddy.json")
        self.merged_file = os.path.join(self.output_dir, "merged_raw.json")
        self.cleaned_file = os.path.join(self.output_dir, "cleaned_corpus.json")

        # Từ điển chuẩn hóa tiếng Việt (Teencode / Viết tắt)
        self.abbreviations = {
            r'\bbsi\b': 'bác sĩ',
            r'\bbs\b': 'bác sĩ',
            r'\bko\b': 'không',
            r'\bđc\b': 'được',
            r'\bsp\b': 'sản phẩm',
            r'\bvs\b': 'với',
            r'\bnc\b': 'nước'
        }

        # Các mẫu câu Call-to-action (CTA) cần loại bỏ
        self.cta_patterns = [
            r'liên hệ ngay( hotline| số điện thoại)?.*?\d+',
            r'gọi ngay.*?\d+',
            r'truy cập( ngay)? website.*',
            r'để lại số điện thoại.*',
            r'mua hàng tại( link)?.*',
            r'chi tiết liên hệ.*'
        ]

    def _generate_hash(self, title: str, content: str) -> str:
        """Tạo mã băm MD5 dựa trên title và content để phát hiện trùng lặp."""
        text_to_hash = f"{title}_{content}".encode('utf-8')
        return hashlib.md5(text_to_hash).hexdigest()

    def consolidate_data(self) -> List[Dict[str, Any]]:
        """Bước 1.1: Đọc, hợp nhất dữ liệu và loại bỏ trùng lặp."""
        logger.info("Bắt đầu hợp nhất dữ liệu (Data Consolidation)...")
        
        merged_data = []
        seen_hashes = set()
        duplicates_count = 0

        # Hàm helper để đọc file và gán source
        def _process_file(filepath: str, source_name: str):
            nonlocal duplicates_count
            if not os.path.exists(filepath):
                logger.warning(f"Không tìm thấy file: {filepath}")
                return

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    title = str(item.get('title', '')).strip()
                    content = str(item.get('content', '')).strip()
                    
                    # Bỏ qua nếu dữ liệu rỗng
                    if not title and not content:
                        continue
                        
                    # Deduplication
                    item_hash = self._generate_hash(title, content)
                    if item_hash in seen_hashes:
                        duplicates_count += 1
                        continue
                        
                    seen_hashes.add(item_hash)
                    
                    # Thêm metadata
                    item['source'] = source_name
                    merged_data.append(item)

        _process_file(self.file_2viet, "2viet")
        _process_file(self.file_papddy, "papddy")

        # Lưu file merged_raw.json
        with open(self.merged_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Hợp nhất hoàn tất. Tổng số bản ghi: {len(merged_data)}.")
        logger.info(f"Đã loại bỏ {duplicates_count} bản ghi trùng lặp.")

        return merged_data

    def _clean_text(self, text: str) -> str:
        """Hàm helper để làm sạch một đoạn văn bản (content hoặc title)."""
        if not isinstance(text, str):
            return ""

        # 1. Chuẩn hóa bảng mã Unicode (Đưa về Unicode dựng sẵn NFC)
        text = unicodedata.normalize('NFC', text)

        # 2. Xóa mã HTML markup
        text = re.sub(r'<[^>]+>', ' ', text)

        # 3. Xóa các cú pháp Markdown rác (VD: hình ảnh, link)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text) # Ảnh markdown
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text) # Lấy text từ link markdown
        text = re.sub(r'[*_#`]', '', text) # Xóa các ký tự format in đậm, in nghiêng

        # 4. Xóa nội dung quảng cáo (CTA)
        for pattern in self.cta_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 5. Chuẩn hóa từ viết tắt/teencode
        for pattern, replacement in self.abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 6. Loại bỏ khoảng trắng thừa, dấu câu lặp lại
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()

    def preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Bước 1.2: Tiền xử lý, chuẩn hóa tiếng Việt và làm sạch rác."""
        logger.info("Bắt đầu tiền xử lý và làm sạch dữ liệu (Data Preprocessing)...")
        
        cleaned_data = []
        for item in data:
            # Làm sạch title và content
            cleaned_title = self._clean_text(item.get('title', ''))
            cleaned_content = self._clean_text(item.get('content', ''))
            
            # Chuẩn hóa tag (lowercase, bỏ khoảng trắng thừa)
            raw_tag = item.get('tag', '')
            if isinstance(raw_tag, list):
                cleaned_tag = [str(t).lower().strip() for t in raw_tag]
            else:
                cleaned_tag = str(raw_tag).lower().strip()

            # Chỉ giữ lại các bản ghi có nội dung đủ dài (ví dụ > 50 ký tự) sau khi làm sạch
            if len(cleaned_content) > 50:
                cleaned_data.append({
                    "url": item.get('url', ''),
                    "title": cleaned_title,
                    "content": cleaned_content,
                    "tag": cleaned_tag,
                    "source": item.get('source', '')
                })

        # Lưu output
        with open(self.cleaned_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Làm sạch hoàn tất. Giữ lại {len(cleaned_data)} bản ghi hợp lệ.")
        return cleaned_data

    def exploratory_data_analysis(self, data: List[Dict[str, Any]]):
        """Bước 1.3: Khám phá dữ liệu (EDA) và xuất báo cáo biểu đồ."""
        logger.info("Bắt đầu phân tích dữ liệu (EDA)...")
        
        if not data:
            logger.error("Không có dữ liệu để thực hiện EDA.")
            return

        df = pd.DataFrame(data)

        # --- 1. Phân tích phân bố Tag ---
        # Xử lý trường hợp tag là list hoặc string
        if isinstance(df['tag'].iloc[0], list):
            df_tags = df.explode('tag')
        else:
            df_tags = df.copy()
            
        plt.figure(figsize=(12, 6))
        tag_counts = df_tags['tag'].value_counts().head(20) # Top 20 tags
        
        # START WWARNING 
        # sns.barplot(x=tag_counts.values, y=tag_counts.index, palette="viridis")
        # END WARNING
        
        sns.barplot(x=tag_counts.values, y=tag_counts.index, hue=tag_counts.index, palette="viridis", legend=False)
        plt.title('Top 20 Tags Phổ Biến Nhất')
        plt.xlabel('Số lượng bài viết')
        plt.ylabel('Tag')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'eda_tag_distribution.png'))
        plt.close()
        
        # --- 2. Phân tích độ dài Token (Ước lượng qua số từ) ---
        # Note: Dùng proxy là số từ (words) chia cho 0.75 để ước lượng số token của Llama 
        # (Vì tiếng Việt thường tốn nhiều token hơn tiếng Anh)
        df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))
        df['estimated_tokens'] = (df['word_count'] / 0.75).astype(int)

        plt.figure(figsize=(10, 5))
        sns.histplot(df['estimated_tokens'], bins=50, kde=True, color='blue')
        plt.title('Phân bố độ dài Token (Ước lượng) của các bài viết')
        plt.xlabel('Số Token')
        plt.ylabel('Tần suất')
        
        # Thêm các đường thống kê
        plt.axvline(df['estimated_tokens'].mean(), color='r', linestyle='--', label=f"Mean: {df['estimated_tokens'].mean():.0f}")
        plt.axvline(df['estimated_tokens'].median(), color='g', linestyle='-', label=f"Median: {df['estimated_tokens'].median():.0f}")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'eda_token_length.png'))
        plt.close()

        # In báo cáo tóm tắt
        logger.info("=== BÁO CÁO TÓM TẮT EDA ===")
        logger.info(f"- Tổng số bài viết: {len(df)}")
        logger.info(f"- Nguồn dữ liệu: \n{df['source'].value_counts().to_string()}")
        logger.info(f"- Thống kê Tokens: Min={df['estimated_tokens'].min()}, Max={df['estimated_tokens'].max()}, Mean={df['estimated_tokens'].mean():.0f}")
        logger.info(f"-> Biểu đồ đã được lưu tại {self.output_dir}")

    def run(self):
        """Hàm thực thi toàn bộ pipeline."""
        logger.info("🚀 BẮT ĐẦU CHẠY PIPELINE DATA ENGINEERING 🚀")
        
        # 1.1 Hợp nhất
        merged_data = self.consolidate_data()
        
        # 1.2 Tiền xử lý
        cleaned_data = self.preprocess_data(merged_data)
        
        # 1.3 EDA
        self.exploratory_data_analysis(cleaned_data)
        
        logger.info("✅ HOÀN TẤT GIAI ĐOẠN 1!")

# Nếu chạy trực tiếp file này
if __name__ == "__main__":
    # Yêu cầu cài đặt các thư viện: pip install pandas matplotlib seaborn
    pipeline = PetDataPipeline(data_dir="./data/raw/", output_dir="./data/processed")
    pipeline.run()
