# 📖 Hướng Dẫn Chạy & Huấn Luyện (Run Guide) - Lab 1: NEU-CLS bằng MLP

Đây là tài liệu đính kèm đáp ứng mục tiêu **Hướng dẫn chạy code (Run README)** thuộc đồ án Lab 1 – Training & Regularization của môn CSC4005. 
Bài thực hành mô phỏng lại việc giải quyết bộ dữ liệu NEU Surface Defect Database để phân loại 6 lỗi bề mặt thép đặc trưng.

---

## 1. Môi trường & Cài đặt

Bạn sẽ cần cài đặt Miniconda/Anaconda cơ bản. Các bước tiếp cận từ gốc:

```bash
# Kích hoạt hoặc thiết lập môi trường ảo ban đầu của bạn (ví dụ deep_learning)
conda activate deep_learning

# Update nền tảng và nạp thư viện quy định
pip install --upgrade pip
pip install -r requirements.txt

# Cài thêm 2 thư viện phụ phục vụ báo cáo và vẽ đồ thị (quan trọng)
pip install wandb matplotlib
```

*(Lưu ý: Nếu bạn muốn đồng bộ biểu đồ huấn luyện như trong bảng báo cáo, nhớ chạy thêm `wandb login` tại Terminal để xác thực API Key của hệ thống online trước khi ấn chạy lệnh).*

---

## 2. Quản Lý Dữ Liệu (NEU-CLS Dataset)

Giống như thông lệ Git thông thường, thư mục này không đính kèm file ảnh Dataset nặng nề. Do đó, bạn cần:
- Sở hữu tệp `NEU-CLS.zip` hoặc thư mục ảnh đã giải nén trên máy trạm của bạn.
- Truyền thẳng đường dẫn tuyệt đối của nó cho hàm `--data_dir`. Hệ thống sẽ **tự động giải nén** và đưa vào DataLoader!

Ví dụ đường dẫn mẫu được dùng trong các báo cáo cho máy: `"d:\DeepLearning\DL\csc4005-lab1-neu-mlp-tyanzuq2811\NEU-CLS.zip"`

---

## 3. Các Lệnh Huấn Luyện Chính (Training Commands)

Bên dưới là các lệnh đã được cấu hình chặt chẽ để thi hành logic `src/train.py`. Để theo dõi lý luận tại sao lại chia các cấu hình tinh chỉnh như dưới đây, bạn hãy tham khảo file **`REPORT.md`**.

### Lệnh Mốc: Chạy mô hình Baseline tiêu chuẩn
Đây là cấu hình chuẩn hóa với AdamW, dùng làm điểm neo (Baseline) đánh giá hiệu suất.
```bash
python -m src.train --data_dir "NEU-CLS.zip" --project csc4005-lab1-neu-mlp --run_name baseline_adamw --optimizer adamw --lr 0.001 --weight_decay 0.0001 --dropout 0.3 --epochs 20 --batch_size 32 --img_size 64 --patience 5 --augment --use_wandb
```

### Lệnh Phá Kỷ Lục: Run F (Best Config)
Đây là cấu hình cao nhất cho ra Test Accuracy **55.93%**, đạt chóp giới hạn của mạng chuẩn MLP. (Kết hợp mở rộng số Tế bào nơ-ron lên `1024-512-128`, kết hợp SGD truyền động chậm gán với LR cất cánh cao `0.01` qua Auto LRScheduler):
```bash
python -m src.train --data_dir "NEU-CLS.zip" --project csc4005-lab1-neu-mlp --run_name run_f_hyper_sgd --optimizer sgd --lr 0.01 --weight_decay 0.0 --dropout 0.3 --hidden_dims 1024 512 128 --epochs 50 --scheduler plateau --batch_size 32 --img_size 64 --patience 10 --augment --use_wandb
```

*(Lưu ý: Bạn bắt buộc phải đổi đuôi `"NEU-CLS.zip"` thành đường dẫn nằm trên máy thật của bạn để Script không xả lỗi File Not Found nhé).*

---

## 4. Kiểm thử kết quả trực quan (Testing & Plots)

Trong file core `src/train.py`, hệ thống tự động xuất bộ test cho Test Set khi vòng lặp dừng. 
Tham số trọng số lưu ở đỉnh cao (`best_model.pt`), dữ liệu Log `history.csv`, đường cong Loss (`curves.png`), ma trận nhầm lẫn (`confusion_matrix.png`) và Json Report đều nằm sẵn trong đường viền:
```text
outputs/<run_name>/
```

### Sinh ảnh kiểm định (Đúng / Sai)
Để bứt phá thêm 1 yêu cầu nhỏ của Lab về trực quan hóa ảnh bị phán quyết sai hoặc chỉ định trúng từ tập Test: Hãy kích hoạt script ngoại vi (Được hard-code theo Output của Best Model - Run F):
```bash
python plot_examples.py
```
Nó sẽ rà trúng file Tensor Model hiện tại và in tấm ảnh `predictions.png` vào trong chính thư mục `outputs/run_f_hyper_sgd/`.

---

## 5. Danh mục các link đánh giá 
- **`REPORT.md`**: File Đọc Toàn Bộ Thí Nghiệm (Có đầy đủ lý lẽ Overfitting / Underfitting, kết cấu mô hình và trích xuất Validation khoa học).
- **`Trang W&B Monitoring`**: [Theo dõi Lệnh chạy Dashboard tại nền tảng Web](https://wandb.ai/models-dai-nam-university/csc4005-lab1-neu-mlp)

