import json
import os
import random
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAugmentationPipeline:
    def __init__(self):
        # Đường dẫn tới các file dữ liệu hiện tại của bạn
        self.qa_file = "data/distillation/filtered_qa.json"
        self.corpus_file = "data/processed/cleaned_corpus.json"
        
        # Thư mục và file đầu ra mới (Không ghi đè file cũ)
        self.output_dir = "data/distillation/augmented"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.final_train_file = os.path.join(self.output_dir, "final_augmented_train.jsonl")
        self.final_test_file = os.path.join(self.output_dir, "final_benchmark.json")

    def _format_llama3(self, system_prompt: str, user_prompt: str, assistant_response: str) -> str:
        """Định dạng chuỗi văn bản theo đúng chuẩn ChatML của Llama-3-Instruct."""
        text_formatted = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_response}<|eot_id|>"
        )
        return json.dumps({"text": text_formatted}, ensure_ascii=False)

    def run(self):
        logger.info("Bắt đầu quá trình Gộp & Tăng cường dữ liệu (Data Augmentation)...")

        # 1. Đọc dữ liệu
        try:
            with open(self.qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Không tìm thấy file: {e}")
            return

        logger.info(f"Đã tải {len(qa_data)} mẫu QA và {len(corpus_data)} bài viết gốc.")

        # 2. Cố định seed để chia Train/Test luôn nhất quán với kết quả 1003/178 của bạn
        random.seed(42)
        random.shuffle(qa_data)
        
        split_idx = 1003  # Dựa theo con số bạn cung cấp
        train_qa = qa_data[:split_idx]
        test_qa = qa_data[split_idx:]

        augmented_train_data = []

        # === TÁC VỤ 1: RAG (Open-book QA) ===
        # Dạy model cách đọc hiểu 'Thông tin tham khảo' và trả lời không bịa đặt.
        sys_rag = "Bạn là trợ lý AI chuyên gia thú y tận tâm. Hãy trả lời câu hỏi dựa trên thông tin tham khảo được cung cấp một cách chính xác và đồng cảm. Tuyệt đối không tự bịa đặt."
        for item in train_qa:
            user_prompt = f"Thông tin tham khảo:\n{item.get('original_content', '')}\n\nCâu hỏi: {item['question']}"
            augmented_train_data.append(self._format_llama3(sys_rag, user_prompt, item['answer']))

        # === TÁC VỤ 2: Closed-book QA ===
        # Dạy model kiến thức nền tảng, có thể tự trả lời ngay cả khi hệ thống VectorDB tìm không ra ngữ cảnh.
        sys_general = "Bạn là một chuyên gia thú y giàu kinh nghiệm và tận tâm. Hãy giải đáp thắc mắc của người dùng một cách chi tiết, dễ hiểu và đưa ra lời khuyên hữu ích."
        for item in train_qa:
            user_prompt = item['question']
            augmented_train_data.append(self._format_llama3(sys_general, user_prompt, item['answer']))

        # === TÁC VỤ 3: Tóm tắt & Viết bài (Knowledge Internalization) ===
        # Tận dụng luôn 300 bài viết thô (cleaned_corpus.json) để dạy model văn phong tiếng Việt tự nhiên và sâu sắc.
        for item in corpus_data:
            user_prompt = f"Hãy chia sẻ cho tôi một số thông tin và kiến thức y khoa thú y chi tiết về chủ đề: {item['title']}"
            augmented_train_data.append(self._format_llama3(sys_general, user_prompt, item['content']))

        # 3. Trộn đều toàn bộ siêu tập dữ liệu
        random.shuffle(augmented_train_data)

        # 4. Lưu ra file JSONL mới
        with open(self.final_train_file, 'w', encoding='utf-8') as f:
            for line in augmented_train_data:
                f.write(line + '\n')

        # 5. Lưu file Benchmark (Giữ nguyên định dạng list JSON để sau này tiện cho đánh giá)
        with open(self.final_test_file, 'w', encoding='utf-8') as f:
            json.dump(test_qa, f, ensure_ascii=False, indent=4)

        logger.info("✅ HOÀN TẤT TĂNG CƯỜNG DỮ LIỆU!")
        logger.info(f"-> File Train mới: {self.final_train_file} (Tổng cộng {len(augmented_train_data)} mẫu)")
        logger.info(f"-> File Test mới: {self.final_test_file} (Tổng cộng {len(test_qa)} mẫu)")

if __name__ == "__main__":
    pipeline = DataAugmentationPipeline()
    pipeline.run()
