import json
import os
import asyncio
import logging
import random
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# Load biến môi trường
load_dotenv()

# Thiết lập Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeDistillationPipeline:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Files đầu ra
        self.raw_qa_file = os.path.join(self.output_dir, "raw_qa.json")
        self.filtered_qa_file = os.path.join(self.output_dir, "filtered_qa.json")
        self.train_file = os.path.join(self.output_dir, "train_ready.jsonl")
        self.test_file = os.path.join(self.output_dir, "benchmark_dataset.json")
        
        # Khởi tạo Async OpenAI client
        api_key = os.getenv("OPENAI_API_KEY", "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")
        if not api_key:
            raise ValueError("Vui lòng thiết lập OPENAI_API_KEY trong file .env")
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Giảm Semaphore xuống 5 để tránh Rate Limit (Too many requests)
        self.semaphore = asyncio.Semaphore(5) 
        self.model_name = "gpt-4o-mini"
        self.max_retries = 5 # Số lần thử lại tối đa

    async def _call_api_with_retry(self, messages: list, temperature: float = 0.7) -> Dict:
        """Hàm gọi API có cơ chế tự động thử lại (Exponential Backoff) khi gặp lỗi 429."""
        base_delay = 2
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    response_format={ "type": "json_object" },
                    messages=messages,
                    temperature=temperature
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str or "rate_limit" in error_str:
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit. Thử lại sau {wait_time:.1f}s (Lần {attempt+1}/{self.max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Lỗi API (Không phải Rate Limit): {str(e)}")
                    if attempt == self.max_retries - 1:
                        return None
                    await asyncio.sleep(base_delay)
        logger.error("Đã vượt quá số lần thử lại. Bỏ qua sample này.")
        return None

    # --- 2.1 DATA AUGMENTATION (TẠO DỮ LIỆU) ---
    def _get_persona_prompts(self) -> Dict[str, str]:
        """Định nghĩa 4 Personas để đa dạng hóa câu hỏi."""
        return {
            "direct": """Bạn là chuyên gia tạo dữ liệu. Dựa vào nội dung cung cấp, hãy tạo 2 cặp Hỏi-Đáp (QA) trực tiếp. 
                         Câu hỏi rõ ràng. Câu trả lời chính xác, dễ hiểu, thân thiện. 
                         Định dạng JSON: {"qa_pairs": [{"question": "...", "answer": "..."}]}""",
                         
            "panic": """Bạn là chuyên gia tạo dữ liệu. Dựa vào nội dung, hãy tạo 1 cặp QA mô phỏng người dùng đang hoảng loạn, 
                        thú cưng đang bị bệnh. Câu hỏi có thể viết tắt, không dấu, tỏ vẻ lo lắng. 
                        Câu trả lời của Bot: Trấn an -> Hướng dẫn dựa trên nội dung -> Khuyên đi khám thú y.
                        Định dạng JSON: {"qa_pairs": [{"question": "...", "answer": "..."}]}""",
                        
            "negative": """Bạn là chuyên gia tạo dữ liệu. Dựa vào nội dung, hãy tạo 1 cặp QA mà người dùng yêu cầu 
                           kê đơn thuốc cụ thể hoặc hỏi những thứ không thể chẩn đoán online. 
                           Câu trả lời của Bot: Xin lỗi, giải thích lý do không thể kê đơn online an toàn, khuyên đi thú y.
                           Định dạng JSON: {"qa_pairs": [{"question": "...", "answer": "..."}]}"""
        }

    async def _generate_qa_for_chunk(self, chunk: Dict[str, Any], persona_name: str, system_prompt: str) -> List[Dict]:
        """Gọi API tạo QA cho 1 bài viết theo 1 Persona."""
        async with self.semaphore:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"NỘI DUNG GỐC:\n{chunk['content']}"}
            ]
            
            result = await self._call_api_with_retry(messages, temperature=0.7)
            
            if result is None:
                return []
                
            qa_pairs = result.get("qa_pairs", [])
            
            # Gắn thêm metadata
            for qa in qa_pairs:
                qa["source_id"] = chunk.get("url", "unknown")
                qa["persona"] = persona_name
                qa["original_content"] = chunk["content"] # Giữ lại để đối chiếu
            return qa_pairs

    async def run_generation(self):
        """Bước 2.1: Chạy đa luồng tạo dữ liệu."""
        logger.info("Bắt đầu Sinh dữ liệu (Data Augmentation)...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        prompts = self._get_persona_prompts()
        tasks = []
        
        # Lên lịch tất cả các tasks (300 samples * 3 personas)
        for chunk in data:
            for p_name, p_prompt in prompts.items():
                tasks.append(self._generate_qa_for_chunk(chunk, p_name, p_prompt))

        # Thực thi song song có thanh tiến trình
        results = await tqdm.gather(*tasks, desc="Đang sinh QA Pairs")
        
        # Flatten list of lists
        all_qa = [item for sublist in results for item in sublist]
        
        with open(self.raw_qa_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Đã tạo thành công {len(all_qa)} cặp QA thô.")
        return all_qa

    # --- 2.2 LLM-AS-A-FILTER (KIỂM ĐỊNH) ---
    async def _evaluate_qa(self, qa_pair: Dict) -> Dict:
        """Sử dụng LLM để chấm điểm chất lượng cặp QA."""
        async with self.semaphore:
            prompt = f"""Đánh giá cặp Hỏi-Đáp sau dựa trên NỘI DUNG GỐC.
NỘI DUNG GỐC: {qa_pair['original_content']}
---
CÂU HỎI: {qa_pair['question']}
CÂU TRẢ LỜI: {qa_pair['answer']}
---
Tiêu chí:
1. Faithfulness: Trả lời không bịa đặt ngoài nội dung gốc.
2. Tone: Tự nhiên, không giống máy móc (KHÔNG dùng từ "Dựa theo văn bản...").
Chấm điểm từ 1 đến 10. Trả về JSON: {{"score": 8, "reason": "..."}}"""

            messages = [{"role": "user", "content": prompt}]
            eval_result = await self._call_api_with_retry(messages, temperature=0.0)
            
            if eval_result is None:
                # Nếu lỗi API liên tục, đánh dấu 0
                qa_pair["quality_score"] = 0
                qa_pair["eval_reason"] = "API Error"
            else:
                qa_pair["quality_score"] = eval_result.get("score", 0)
                qa_pair["eval_reason"] = eval_result.get("reason", "")
                
            return qa_pair

    async def run_filtering(self, raw_qa: List[Dict]):
        """Bước 2.2: Lọc bỏ các QA chất lượng kém."""
        logger.info("Bắt đầu Kiểm định chất lượng (Filtering)...")
        tasks = [self._evaluate_qa(qa) for qa in raw_qa]
        evaluated_qa = await tqdm.gather(*tasks, desc="Đang chấm điểm QA")

        # Lọc các mẫu đạt điểm >= 7 (Nới lỏng 1 chút để có đủ dữ liệu)
        good_qa = [qa for qa in evaluated_qa if qa["quality_score"] >= 7]
        
        # Post-processing: Sửa lỗi dùng từ của AI
        for qa in good_qa:
            qa["answer"] = qa["answer"].replace("Dựa vào thông tin được cung cấp, ", "")
            qa["answer"] = qa["answer"].replace("Dựa theo văn bản, ", "")
            qa["answer"] = qa["answer"].strip()

        with open(self.filtered_qa_file, 'w', encoding='utf-8') as f:
            json.dump(good_qa, f, ensure_ascii=False, indent=4)
            
        logger.info(f"Đã lọc xong. Giữ lại {len(good_qa)}/{len(raw_qa)} QA chất lượng cao.")
        return good_qa

    # --- 2.3 FORMATTING Llama 3 ---
    def format_to_llama3(self, qa_data: List[Dict]):
        """Bước 2.3: Định dạng theo chuẩn ChatML của Llama 3 và chia Train/Test."""
        logger.info("Bắt đầu định dạng dữ liệu cho Llama 3...")
        
        # Trộn ngẫu nhiên dữ liệu
        random.shuffle(qa_data)
        
        # Chia 85% Train, 15% Test
        split_idx = int(len(qa_data) * 0.85)
        train_data = qa_data[:split_idx]
        test_data = qa_data[split_idx:]

        system_prompt = "Bạn là trợ lý AI chuyên gia thú y tận tâm. Hãy trả lời các câu hỏi về chăm sóc, sức khỏe thú cưng một cách chính xác, đồng cảm và luôn khuyên người dùng đưa thú cưng đến trạm thú y nếu có dấu hiệu bệnh nặng."

        # Ghi file Train (.jsonl) chuẩn Llama 3
        with open(self.train_file, 'w', encoding='utf-8') as f:
            for item in train_data:
                # Cấu trúc chuỗi ChatML chính xác cho Llama-3-Instruct
                text_formatted = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{item['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    f"{item['answer']}<|eot_id|>"
                )
                
                # Unsloth/HuggingFace thường nhận field 'text'
                json_line = json.dumps({"text": text_formatted}, ensure_ascii=False)
                f.write(json_line + '\n')

        # Ghi file Benchmark Test (.json) giữ nguyên metadata
        with open(self.test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Đã tạo file Train: {len(train_data)} mẫu.")
        logger.info(f"Đã tạo file Test/Benchmark: {len(test_data)} mẫu.")

async def main():
    # Khởi tạo pipeline
    pipeline = KnowledgeDistillationPipeline(
        input_file="data/processed/cleaned_corpus.json",
        output_dir="data/distillation"
    )
    
    # Chạy Step 2.1
    raw_qa = await pipeline.run_generation()
    
    # Chạy Step 2.2
    filtered_qa = await pipeline.run_filtering(raw_qa)
    
    # Chạy Step 2.3
    pipeline.format_to_llama3(filtered_qa)
    
    logger.info("✅ HOÀN TẤT GIAI ĐOẠN 2!")

if __name__ == "__main__":
    # Yêu cầu cài đặt: pip install openai python-dotenv tqdm
    asyncio.run(main())