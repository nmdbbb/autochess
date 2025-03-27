 ♟️ AlphaZero-Chess — AI học chơi cờ vua từ tự chơi (AlphaZero-style)

Dự án này mô phỏng cách AlphaZero của DeepMind học chơi cờ vua **từ con số 0** — không dùng dữ liệu người chơi, không luật cố định — chỉ qua quá trình **tự chơi** và **tối ưu hóa bằng deep reinforcement learning**.

---

## 🚀 Mục tiêu dự án

- ✅ Huấn luyện một AI chơi cờ vua **tự học từ đầu**.
- ✅ Dùng **Monte Carlo Tree Search (MCTS)** để khám phá nước đi.
- ✅ Kết hợp với **mạng ResNet** để ước lượng nước đi và giá trị ván cờ.
- ✅ Tự chơi liên tục, huấn luyện nhiều iteration để cải thiện năng lực.
- ✅ So sánh với Stockfish và cung cấp giao diện chơi thử qua Web.

---

## 🧠 Thuật toán tổng quan

1. **Self-play**: AI tự chơi với chính nó, lưu lại dữ liệu (trạng thái, policy, kết quả).
2. **Train**: Huấn luyện mạng nơ-ron để dự đoán tốt hơn policy và value.
3. **Evaluate**: So sánh model mới với model tốt nhất hiện tại. Nếu tốt hơn → cập nhật.
4. **Repeat**: Quay lại bước 1.

---

## 📂 Cấu trúc thư mục

```
alphazero-chess/
├── main.py
├── train_loop.py
├── selfplay.py
├── train.py
├── evaluate.py
├── evaluate_stockfish.py
├── play_cli.py

├── backend/                  # 🔥 Flask API phục vụ nước đi cho GUI
│   ├── move.py
│   ├── requirements.txt
│   └── Dockerfile

├── api/                      # (tuỳ chọn) API legacy (có thể gộp với backend)

├── web/                      # Giao diện React để chơi với AI
│   └── src/ChessAIPlay.tsx

├── models/
│   ├── resnet.py
│   └── utils.py

├── mcts/
│   └── mcts.py

├── chess_env/
│   └── board.py

├── utils/
│   ├── config.py
│   ├── logger.py
│   └── plot.py

├── data/
│   ├── replay_buffer.pkl
│   └── models/
│       ├── model_000.pth
│       ├── model_001.pth
│       └── model_best.pth

├── config.yaml
├── requirements.txt
├── docker-compose.yml
```

---

## ⚙️ Cài đặt & chạy nhanh

### ✅ Cài Python package

```bash
pip install -r requirements.txt
```

### ✅ Chạy huấn luyện
```bash
python train_loop.py   # Huấn luyện nhiều vòng
```

### ✅ Tự chơi, huấn luyện, đánh giá 1 vòng
```bash
python main.py
```

---

## 🎮 Chơi thử với AI

### Giao diện dòng lệnh:
```bash
python play_cli.py
```

### Giao diện Web:
```bash
# 1. Chạy backend
cd api/
python move.py

# 2. Mở frontend
cd web/
npm install
npm run dev
```

> Truy cập: http://localhost:5173

---

## 🤖 So sánh với Stockfish

```bash
python evaluate_stockfish.py
```

Yêu cầu đã cài `stockfish` trong hệ thống (`apt install stockfish` hoặc [tải tại đây](https://stockfishchess.org/download/)).

---

## 🐳 Dùng Docker Compose

```bash
docker-compose up --build
```

- Giao diện: http://localhost:5173  
- API Flask: http://localhost:5000/api/move

---

## 🛠️ Cấu hình (`config.yaml`)

```yaml
model:
  input_planes: 119
  channels: 256
  residual_blocks: 10

training:
  batch_size: 256
  learning_rate: 0.01
  weight_decay: 1e-4

mcts:
  simulations: 400
  cpuct: 1.5
  temperature: 1.0
  dirichlet_alpha: 0.3

selfplay:
  games_per_iteration: 100
  max_moves: 512
```

---

## 📚 Tham khảo

- [AlphaZero Paper (DeepMind)](https://arxiv.org/abs/1712.01815)
- [Leela Chess Zero](https://github.com/LeelaChessZero/lc0)
- [python-chess](https://python-chess.readthedocs.io)

---

> 👨‍💻 Dự án học thuật: Tự xây dựng AI cờ vua từ self-play và deep learning. Không cần dữ liệu người — AI học từ chính nó.