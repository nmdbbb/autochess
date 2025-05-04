# 📘 Mô tả Dự án: AlphaZero-Chess (Transformer + Distributed RL)

## 🎯 Tên dự án:
**AlphaZero-Chess — Xây dựng AI học chơi cờ vua bằng Reinforcement Learning phân tán với Transformer**

## 💡 Mục tiêu chính:
- Xây dựng một mô hình AI có khả năng **tự học chơi cờ vua từ đầu** mà không cần dữ liệu con người.
- Mô phỏng cách hoạt động của **AlphaZero** của DeepMind:
  - Kết hợp **Monte Carlo Tree Search (MCTS)** với mạng nơ-ron sâu **Transformer**, thay thế ResNet truyền thống.
  - Học chiến lược hoàn toàn qua **self-play**.
  - Huấn luyện song song và phân tán bằng **Ray RLlib** hoặc **PyTorch DDP**.
- Đánh giá bằng cách thi đấu với các engine như Stockfish.
- Hỗ trợ giao diện người dùng Web + CLI để tương tác với AI.

## 🧠 Ý tưởng chính
1. AI bắt đầu từ trắng tay, không có dữ liệu con người.
2. Dùng Transformer để ước lượng xác suất nước đi (policy) và khả năng thắng (value).
3. MCTS sử dụng thông tin từ Transformer để duyệt cây.
4. Self-play sinh dữ liệu (s, π, z) liên tục.
5. Toàn bộ pipeline huấn luyện và sinh dữ liệu được phân tán để tăng tốc.

## ⚙️ Kiến trúc hệ thống
```
[Web UI / CLI] ─▶ [REST API Flask] ─▶ [Self-Play Worker] ─▶ [Transformer + MCTS Engine]
                                          │
                                          ▼
                             [Replay Buffer Distributed]
                                          │
                                          ▼
                           [Trainer Node - Distributed GPUs]
```

## 🧩 Thành phần và công nghệ
| Thành phần       | Công nghệ sử dụng                     |
|------------------|----------------------------------------|
| Neural Network   | PyTorch, Transformer Encoder-Only      |
| Self-Play Engine | MCTS C++, Python, Ray Actor            |
| Luật chơi        | python-chess hoặc C++ Custom Engine    |
| Distributed RL   | Ray RLlib / PyTorch Lightning + DDP    |
| REST API         | Flask, FastAPI                         |
| UI               | ReactJS + TypeScript                   |
| DevOps           | Docker, docker-compose, GitHub Actions|

## 📦 Cấu trúc thư mục
```
alphazero-chess/
├── models/              # Mạng Transformer + weight loader
├── mcts/                # MCTS Engine viết bằng C++ / Python
├── selfplay/            # Ray Actor chạy tự chơi và sinh dữ liệu
├── train/               # Trainer dùng DDP hoặc RLlib
├── backend/             # REST API Flask
├── web/                 # React Frontend
├── config/              # YAML cấu hình pipeline
├── distributed/         # Script chạy cluster phân tán
├── data/                # Replay buffer, checkpoints
├── evaluate.py          # So sánh với Stockfish
├── play_cli.py          # Giao diện chơi bằng dòng lệnh
└── docker-compose.yml   # Triển khai toàn bộ stack
```


## ⚡ Cấu hình đề xuất
| Hạng mục        | Tối thiểu         | Khuyến nghị chuyên sâu |
|-----------------|-------------------|--------------------------|
| GPU             | RTX 3060 12GB     | A100 40GB × N (multi-GPU) |
| CPU             | 8 cores           | ≥32 cores (Ray cluster)  |
| RAM             | 16 GB             | ≥64 GB                   |
| Lưu trữ         | 100GB SSD         | ≥1TB NVMe                |
| Network         | -                 | 1Gbps LAN nếu chạy phân tán |