# Sử dụng Python 3.10 chính thức
FROM python:3.10-slim

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép toàn bộ mã nguồn
COPY . /app

# Cài đặt Python dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Mặc định chạy API Flask để phục vụ GUI
CMD ["python", "api/move.py"]
