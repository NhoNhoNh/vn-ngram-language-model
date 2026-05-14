# Vietnamese N-gram Language Model

Dự án này triển khai mô hình ngôn ngữ N-gram (Bigram, Trigram,...) cho tiếng Việt với các kỹ thuật làm mịn (smoothing) khác nhau và hệ thống đánh giá hiệu năng chi tiết.

## 📺 Demo Video
[![Xem Demo](https://img.youtube.com/vi/omisfP0lVwo/0.jpg)](https://www.youtube.com/watch?v=omisfP0lVwo)

## 📦 Dataset & Model
Bạn có thể tải tập dữ liệu thu thập từ wiki đã xử lý để huấn luyện các mô hình tại Hugging Face:
- **Hugging Face Dataset**: [vietnamese-wiki-ngram](https://huggingface.co/datasets/nhonguyen25/vietnamese-wiki-ngram)

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

### 3. Chạy Web Application (Giao diện người dùng)
Dự án cung cấp một giao diện web đơn giản để bạn trải nghiệm việc gợi ý từ tiếp theo:
```bash
# Cài đặt thêm thư viện cho web
pip install fastapi uvicorn pydantic

# Đảm bảo các file model (.pkl.gz) đã có sẵn trong thư mục gốc
# Chạy ứng dụng
python app.py
```
Sau đó, truy cập `http://127.0.0.1:8000` trên trình duyệt.

## 📊 Kết quả đánh giá sơ bộ

| Model | Perplexity | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :---: | :---: | :---: |
| Trigram (Kneser-Ney) | ~305.11 | ~22.04% | ~38.34% |
| Trigram (Add-k) | ~4125.87 | ~21.41% | ~36.62% |
| Trigram (Laplace) | ~39143.77 | ~21.41% | ~36.62% |

*(Lưu ý: Kết quả thực tế phụ thuộc vào kích thước corpus và tham số k/discount)*

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ**: Python
- **Thư viện NLP**: PyVi, Underthesea
- **Phân tích dữ liệu**: Pandas, Numpy
- **Công cụ**: Jupyter Notebook / Google Colab
