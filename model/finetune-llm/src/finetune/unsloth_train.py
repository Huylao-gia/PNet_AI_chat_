# %% [markdown]
# # GIAI ĐOẠN 3: HUẤN LUYỆN MÔ HÌNH LLAMA-3.2-1B VỚI UNSLOTH (QLoRA)
# **Môi trường yêu cầu:** Google Colab (Runtime: T4 GPU) hoặc Máy ảo Linux có GPU NVIDIA.
# **Lưu ý:** Chạy lần lượt từng Cell (ô code) bên dưới.

# %% [markdown]
# ### BƯỚC 1: Cài đặt thư viện (Chỉ chạy trên Google Colab)
# Bỏ ghi chú (uncomment) và chạy ô này nếu bạn đang dùng Colab.
# Nếu chạy trên máy ảo, hãy chạy các lệnh pip này trong terminal.

# %%
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes
# !pip install wandb datasets

# %% [markdown]
# ### BƯỚC 2: Khai báo thư viện & Đăng nhập Weights & Biases
# Chúng ta dùng wandb để theo dõi biểu đồ Loss (sai số) trong quá trình train, giúp phát hiện sớm Overfitting.

# %%
import torch
import wandb
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# (Tùy chọn) Điền API Key của WandB vào đây, hoặc hệ thống sẽ yêu cầu bạn dán vào khi chạy.
# wandb.login(key="WANDB_API_KEY_CỦA_BẠN_NẾU_CÓ")

# %% [markdown]
# ### BƯỚC 3: Nạp Mô hình gốc (Base Model) & Tokenizer
# Sử dụng Unsloth để nạp mô hình Llama-3.2-1B ở định dạng 4-bit, giúp tiết kiệm tối đa VRAM.

# %%
max_seq_length = 2048 # Độ dài ngữ cảnh tối đa (Tương đương khoảng 1500 từ). Đủ cho RAG.
dtype = None # Unsloth tự động chọn (Bfloat16 cho Ampere+, Float16 cho T4/V100)
load_in_4bit = True # Bắt buộc True để dùng QLoRA (Tiết kiệm VRAM)

print("Đang nạp mô hình Llama-3.2-1B-Instruct...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "phamhai/Llama-3.2-1B-Instruct-Frog",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
print("Nạp mô hình thành công!")

# %% [markdown]
# ### BƯỚC 4: Cấu hình QLoRA Adapters
# Chúng ta không train toàn bộ 1 tỷ tham số, mà chỉ thêm vào các ma trận nhỏ (adapters).
# Rank (r) = 32 giúp mô hình học các khái niệm y khoa phức tạp.

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Mức độ phức tạp của LoRA (Khuyên dùng: 16, 32, 64)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # Áp dụng lên tất cả Attention & MLP
    lora_alpha = 64, # Thường gấp đôi r
    lora_dropout = 0, # Tối ưu hóa (Unsloth khuyên dùng 0)
    bias = "none",    # Tối ưu hóa
    use_gradient_checkpointing = "unsloth", # Cực kỳ quan trọng: Giảm 30% VRAM
    random_state = 42,
    use_rslora = False, 
)

# %% [markdown]
# ### BƯỚC 5: Nạp Tập dữ liệu Huấn luyện
# Đọc file `final_augmented_train.jsonl` (đã được format sẵn chuẩn ChatML ở Giai đoạn 2).

# %%
# Tải file jsonl từ Google Drive hoặc thư mục local lên Colab
# Ví dụ trên Colab: upload file vào thư mục mặc định '/content/final_augmented_train.jsonl'
dataset_file = "final_augmented_train.jsonl" 

dataset = load_dataset("json", data_files=dataset_file, split="train")
print(f"Tổng số mẫu huấn luyện: {len(dataset)}")

# %% [markdown]
# ### BƯỚC 6: Thiết lập tham số Huấn luyện (SFTTrainer)
# Với dữ liệu chất lượng cao, chỉ cần 1 đến 2 epochs là đủ. Train nhiều hơn dễ gây ảo giác (overfit).

# %%
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # Trường chứa nội dung trong file jsonl
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Set True nếu muốn train nhanh hơn cho chuỗi ngắn, False an toàn hơn cho RAG
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Effective Batch Size = 2 * 4 = 8
        warmup_steps = 50, # Khởi động learning rate từ từ
        num_train_epochs = 1, # Số vòng lặp huấn luyện. Bạn có thể tăng lên 2 nếu loss vẫn cao.
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit", # Optimizer 8-bit tiết kiệm VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
        report_to = "wandb", # Báo cáo kết quả lên Weights & Biases
    ),
)

# %% [markdown]
# ### BƯỚC 7: Bắt đầu Huấn luyện 🚀
# Theo dõi cột `Loss`, nếu nó giảm dần dần từ (ví dụ) 2.0 xuống 0.7 - 0.5 là mô hình đang học rất tốt.

# %%
trainer_stats = trainer.train()

# %% [markdown]
# ### BƯỚC 8: Kiểm thử trực tiếp Mô hình sau khi Train (Inference Test)
# Hãy hỏi thử một câu RAG để xem mô hình trả lời thế nào.

# %%
# Đặt mô hình ở chế độ suy luận (nhanh hơn 2x)
FastLanguageModel.for_inference(model) 

system_prompt = "Bạn là trợ lý AI chuyên gia thú y tận tâm. Hãy trả lời câu hỏi dựa trên thông tin tham khảo được cung cấp một cách chính xác và đồng cảm. Tuyệt đối không tự bịa đặt."
context = "Bệnh Care ở chó (Canine Distemper) là bệnh truyền nhiễm nguy hiểm do virus gây ra, ảnh hưởng đến hệ hô hấp, tiêu hóa và thần kinh. Chó chưa tiêm phòng rất dễ mắc bệnh."
question = "Chó nhà tôi chưa tiêm phòng, dạo này thấy ho và tiêu chảy. Liệu có phải bệnh Care không?"

test_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

Thông tin tham khảo:
{context}

Câu hỏi: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer([test_prompt], return_tensors = "pt").to("cuda")

# Sinh câu trả lời
outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
answer = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]

print("\n--- CÂU TRẢ LỜI CỦA MÔ HÌNH ---")
# Cắt bỏ phần prompt ban đầu để chỉ in ra câu trả lời
print(answer.split("assistant")[-1].strip()) 

# %% [markdown]
# ### BƯỚC 9: Xuất Mô hình (GGUF Export)
# Định dạng GGUF (chạy bằng llama.cpp hoặc Ollama) là đỉnh cao của sự tối ưu.
# Nó nén file chỉ còn khoảng ~1GB và có thể chạy cực mượt trên CPU hoặc VPS giá rẻ ở Giai đoạn 4.

# %%
# Lưu model ở định dạng GGUF q4_k_m (Lượng hóa 4-bit cân bằng nhất giữa dung lượng và chất lượng)
model.save_pretrained_gguf("Llama-3.2-1B-Pet-Chatbot-GGUF", tokenizer, quantization_method = "q4_k_m")

print("Đã lưu thành công thư mục: Llama-3.2-1B-Pet-Chatbot-GGUF")
# Bạn hãy tải thư mục này từ Colab về máy tính/VPS để sử dụng cho bước Hosting.
