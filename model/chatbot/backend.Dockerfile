# filepath: backend.Dockerfile
# Sử dụng phiên bản Python 3.10 mỏng nhẹ để tối ưu dung lượng
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong Container
WORKDIR /app

# Khắc phục lỗi thiếu thư viện hệ thống khi cài đặt một số package Python
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt thư viện Python trước (Tận dụng cache của Docker để build nhanh hơn)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn Backend vào Container
COPY backend/ /app/

# Khai báo cổng 8080 cho API
EXPOSE 8080

# Lệnh khởi chạy server bằng Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
