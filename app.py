from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
from pyvi import ViTokenizer

# Import class mô hình mà bạn đã viết
from ngram_model import NgramLanguageModel

app = FastAPI()

# Biến toàn cục chứa các mô hình
MODELS = {}

def load_all_models():
    """Tải các mô hình N-gram từ ổ cứng lên RAM"""
    print("Đang tải các mô hình, vui lòng đợi...")
    
    # Định nghĩa các file model Trigram (n=3)
    model_files = {
        'laplace_3': 'trigram_laplace.pkl.gz',
        'kn_3': 'trigram_kn.pkl.gz',
        'addk_3': 'trigram_add_k.pkl.gz',
    }
    
    for key, filename in model_files.items():
        if os.path.exists(filename):
            MODELS[key] = NgramLanguageModel.load(filename, compressed=True)
            print(f"  -> Đã load {filename} thành công!")
        else:
            print(f"  -> Cảnh báo: Không tìm thấy {filename}")
            
    print("Hoàn tất tải mô hình!")

@app.on_event("startup")
async def startup_event():
    load_all_models()

class SuggestRequest(BaseModel):
    text: str
    method: str = 'kn'  # Mặc định dùng Kneser-Ney
    top_k: int = 5

@app.post("/suggest")
async def suggest_words(req: SuggestRequest):
    results = {"suggestions": []}
    text = req.text.strip()
    
    if not text:
        return results

    # 1. Tokenize bằng PyVi để khớp với định dạng lúc train (nối từ bằng dấu '_')
    tokens = ViTokenizer.tokenize(text.lower()).split()
    
    if not tokens:
        return results

    # 2. Cố định sử dụng Trigram (n=3)
    n = 3
    model_key = f"{req.method}_{n}"
    
    if model_key not in MODELS:
        return results
        
    model = MODELS[model_key]
    
    # 3. Lấy context và padding <START> nếu text quá ngắn
    ctx_len = n - 1
    if len(tokens) >= ctx_len:
        context = tuple(tokens[-ctx_len:])
    else:
        pad_len = ctx_len - len(tokens)
        context = tuple(['<START>'] * pad_len + tokens)
        
    # 4. Lấy dự đoán từ mô hình
    predictions = model.predict_next(context, top_k=req.top_k)
    
    # 5. Format lại output (bỏ dấu gạch dưới '_' để hiển thị trên UI đẹp hơn)
    results["suggestions"] = [
        {"word": word.replace('_', ' '), "prob": round(prob * 100, 4)} 
        for word, prob in predictions
    ]
    
    return results

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    if not os.path.exists("templates"):
        os.makedirs("templates")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)