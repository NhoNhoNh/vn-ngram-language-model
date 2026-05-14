import pickle
import gzip
import logging
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Union
from tqdm.auto import tqdm

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class NgramLanguageModel:
    """
    Mô hình N-gram Language Model thống kê truyền thống.
    Hỗ trợ N-gram tổng quát (Bigram, Trigram...) và nhiều chiến lược Smoothing.
    """

    def __init__(self, n: int = 3, smoothing: str = 'laplace', k: float = 1.0, discount: float = 0.75):
        """
        :param n: Kích thước cửa sổ N-gram (2 cho Bigram, 3 cho Trigram).
        :param smoothing: Phương pháp làm mịn ('none', 'laplace', 'add-k', 'kneser-ney').
        :param k: Tham số k cho Add-k (Laplace mặc định k=1.0).
        :param discount: Tham số d (discount) cho Kneser-Ney (thường từ 0.5 đến 0.75).
        """
        assert n >= 2, "Model cần ít nhất n=2 (Bigram)"
        assert smoothing in ['none', 'laplace', 'add-k', 'kneser-ney'], "Smoothing không hợp lệ!"

        self.n = n
        self.smoothing = smoothing
        self.k = 1.0 if smoothing == 'laplace' else k
        self.discount = discount

        # Bảng tần suất
        self.vocab = set()
        self.vocab_size = 0

        # self.counts[context][target] = count
        self.counts = defaultdict(Counter)

        # self.context_totals[context] = tổng số lần context xuất hiện
        self.context_totals = Counter()

        # --- Dành riêng cho Kneser-Ney ---
        self.continuation_counts = Counter()
        self.total_ngram_types = 0

    def fit(self, corpus: List[List[str]], vocab: set):
        """
        Huấn luyện mô hình bằng cách đếm tần suất N-gram từ corpus.
        """
        logger.info(f"Đang huấn luyện {self.n}-gram model với smoothing '{self.smoothing}'...")
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        for sentence in tqdm(corpus, desc="Counting N-grams"):
            if len(sentence) < self.n:
                continue

            # Trượt cửa sổ N-gram qua từng câu
            for i in range(len(sentence) - self.n + 1):
                window = sentence[i : i + self.n]
                context = tuple(window[:-1])
                target = window[-1]

                self.counts[context][target] += 1
                self.context_totals[context] += 1

                if self.smoothing == 'kneser-ney':
                    # Đếm số lượng context khác nhau đứng trước target w
                    # Cần thiết để tính P_continuation
                    self.continuation_counts[target] += 1
                    self.total_ngram_types += 1

        # Memory Optimization: Convert defaultdict về dict chuẩn để giảm RAM
        logger.info("Đang tối ưu hóa bộ nhớ (Freezing dicts)...")
        self.counts = {ctx: dict(target_counts) for ctx, target_counts in self.counts.items()}
        self.context_totals = dict(self.context_totals)

        if self.smoothing == 'kneser-ney':
            self.continuation_counts = dict(self.continuation_counts)

        logger.info(f"Huấn luyện xong! Ghi nhận {len(self.counts):,} unique contexts.")

    def predict_proba(self, context: Tuple[str, ...], target: str) -> float:
        """
        Tính xác suất P(target | context).
        """
        # Fallback an toàn nếu context hoặc target chứa từ không có trong vocab
        safe_target = target if target in self.vocab else '<UNK>'
        safe_context = tuple([w if w in self.vocab else '<UNK>' for w in context])

        # Nếu context nhập vào dài hơn thiết lập của mô hình, cắt bớt lấy phần đuôi
        if len(safe_context) > self.n - 1:
            safe_context = safe_context[-(self.n - 1):]
        # Nếu context nhập vào ngắn hơn, đệm thêm <START>
        elif len(safe_context) < self.n - 1:
            pad_len = (self.n - 1) - len(safe_context)
            safe_context = tuple(['<START>'] * pad_len) + safe_context

        count_c_w = self.counts.get(safe_context, {}).get(safe_target, 0)
        count_c = self.context_totals.get(safe_context, 0)

        # 1. No Smoothing (Maximum Likelihood)
        if self.smoothing == 'none':
            if count_c == 0: return 0.0
            return count_c_w / count_c

        # 2 & 3. Laplace / Add-k Smoothing
        elif self.smoothing in ['laplace', 'add-k']:
            return (count_c_w + self.k) / (count_c + self.k * self.vocab_size)

        # 4. Kneser-Ney Smoothing (Interpolated)
        elif self.smoothing == 'kneser-ney':
            # Xác suất Continuation
            p_cont = self.continuation_counts.get(safe_target, 0) / max(1, self.total_ngram_types)

            if count_c == 0:
                # Nếu context chưa từng xuất hiện, trả về hoàn toàn xác suất continuation
                return p_cont

            # Tính Discounted Probability
            discounted_prob = max(count_c_w - self.discount, 0) / count_c

            # Tính Lambda (Trọng số nội suy)
            unique_continuations = len(self.counts.get(safe_context, {}))
            lambda_weight = (self.discount / count_c) * unique_continuations

            return discounted_prob + lambda_weight * p_cont

    def predict_next(self, context: Tuple[str, ...], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Dự đoán K từ tiếp theo có xác suất cao nhất.
        Thực hiện brute-force tính xác suất trên toàn bộ Vocabulary.
        """
        probabilities = []
        for word in self.vocab:
            # Bỏ qua token không mang ý nghĩa sinh văn bản
            if word in ['<START>', '<UNK>']:
                continue

            prob = self.predict_proba(context, word)
            probabilities.append((word, prob))

        # Sort giảm dần theo xác suất
        probabilities.sort(key=lambda x: x[1], reverse=True)
        return probabilities[:top_k]

    def save(self, filepath: str, compressed: bool = True):
        """Lưu model. Dùng gzip để nén vì count dictionary rất tốn dung lượng."""
        logger.info(f"Đang lưu model tại {filepath} (Compressed: {compressed})...")
        open_func = gzip.open if compressed else open

        with open_func(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Đã lưu xong!")

    @classmethod
    def load(cls, filepath: str, compressed: bool = True) -> 'NgramLanguageModel':
        """Load model từ file."""
        logger.info(f"Đang load model từ {filepath}...")
        open_func = gzip.open if compressed else open

        with open_func(filepath, 'rb') as f:
            data = pickle.load(f)

        # Khôi phục instance
        model = cls(n=data['n'], smoothing=data['smoothing'], k=data['k'], discount=data['discount'])
        model.__dict__.update(data)

        logger.info("Load model thành công!")
        return model
