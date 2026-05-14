# Vietnamese N-gram Language Model

Dự án này triển khai mô hình ngôn ngữ N-gram (Bigram, Trigram, ...) cho tiếng Việt với các kỹ thuật làm mịn (smoothing) khác nhau và hệ thống đánh giá hiệu năng chi tiết.

## 📌 Tính năng chính

- **Hỗ trợ N-gram tổng quát**: Dễ dàng tùy chỉnh kích thước cửa sổ (n=2, 3, ...).
- **Kỹ thuật làm mịn (Smoothing)**:
  - Laplace Smoothing (Add-one).
  - Add-k Smoothing.
  - Kneser-Ney Smoothing (Interpolated) - Kỹ thuật tối ưu cho mô hình N-gram.
- **Tiền xử lý tiếng Việt chuyên sâu**:
  - Tách từ (Word Segmentation) sử dụng `PyVi` và `underthesea`.
  - Xử lý từ hiếm bằng token `<UNK>`.
  - Hỗ trợ padding `<START>` và `<END>`.
- **Tối ưu hóa hiệu năng**:
  - Tối ưu bộ nhớ bằng cách đóng băng (freezing) dictionary.
  - Lưu trữ mô hình nén bằng `gzip` và `pickle`.
  - Đánh giá Accuracy siêu tốc bằng thuật toán Candidate Pruning.

## 📂 Cấu trúc dự án

```text
├── crawl_data.ipynb      # Thu thập dữ liệu từ Wikipedia
├── tokenization.ipynb    # Tiền xử lý, tách từ và chuẩn bị dataset
├── train_ngrams.ipynb    # Huấn luyện mô hình với các loại smoothing
├── evaluation.ipynb      # Đánh giá Perplexity, Accuracy và Error Analysis
├── data/                 # Thư mục chứa dữ liệu
│   ├── raws/             # Dữ liệu thô
│   └── train/            # Dữ liệu đã xử lý và các mô hình đã lưu
└── README.md             # Hướng dẫn dự án
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường
Dự án yêu cầu các thư viện sau:
```bash
pip install pyvi underthesea pandas numpy tqdm datasets loguru
```

### 2. Quy trình thực hiện
1. **Crawl Data**: Chạy `crawl_data.ipynb` để lấy corpus tiếng Việt.
2. **Preprocessing**: Chạy `tokenization.ipynb` để làm sạch và tách từ. Kết quả sẽ được lưu vào `data/train/`.
3. **Training**: Chạy `train_ngrams.ipynb` để huấn luyện. Bạn có thể chọn các phương pháp smoothing khác nhau.
4. **Evaluation**: Chạy `evaluation.ipynb` để xem báo cáo về Perplexity và Accuracy của từng mô hình.

## 📊 Kết quả đánh giá sơ bộ

| Model | Perplexity | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :---: | :---: | :---: |
| Trigram (Kneser-Ney) | ~305.11 | ~14.2% | ~28.5% |
| Trigram (Add-k) | ~4125.87 | ... | ... |
| Trigram (Laplace) | ~39143.77 | ... | ... |

*(Lưu ý: Kết quả thực tế phụ thuộc vào kích thước corpus và tham số k/discount)*

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ**: Python
- **Thư viện NLP**: PyVi, Underthesea
- **Phân tích dữ liệu**: Pandas, Numpy
- **Công cụ**: Jupyter Notebook / Google Colab

---
Được thực hiện bởi: [Tên của bạn/Team]
