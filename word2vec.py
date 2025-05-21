import gensim 
from gensim.models import Word2Vec 
import nltk 
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import os

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Try to download NLTK resources
try:
    nltk.download('punkt')
    print("Downloaded NLTK punkt tokenizer")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("If you encounter issues with NLTK, try running these commands manually:")
    print("import nltk")
    print("nltk.download('punkt')")

# Load processed data
df = pd.read_csv("data/processed/functions.csv")
print(f"Loaded {len(df)} code samples")

# Chuẩn bị dữ liệu cho Word2Vec 
tokenized_code = [] 
for code in df['processed_code']: 
    tokens = word_tokenize(str(code)) 
    tokenized_code.append(tokens) 

# Huấn luyện mô hình Word2Vec 
w2v_model = Word2Vec( 
    sentences=tokenized_code, 
    vector_size=100, # Kích thước vector 
    window=5, # Kích thước cửa sổ ngữ cảnh 
    min_count=2, # Tối thiểu số lần xuất hiện của từ
    workers=4 # Số luồng 
) 

# Lưu mô hình 
w2v_model.save("models/w2v_code.model") 

# Tạo embedding cho mỗi hàm bằng cách lấy trung bình các vector từ
def create_document_vector(doc_tokens, model): 
    doc_vector = [] 
    for token in doc_tokens: 
        if token in model.wv: 
            doc_vector.append(model.wv[token]) 
  
    if not doc_vector: 
        return np.zeros(model.vector_size) 
    return np.mean(doc_vector, axis=0) 

# Tạo embedding cho mỗi hàm 
doc_vectors = [] 
for tokens in tokenized_code: 
    doc_vectors.append(create_document_vector(tokens, w2v_model))

# Lưu các vector 
doc_vectors_array = np.array(doc_vectors) 
np.save("data/processed/w2v_vectors.npy", doc_vectors_array) 
print(f"Shape of Word2Vec embeddings: {doc_vectors_array.shape}")

