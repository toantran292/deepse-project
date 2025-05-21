import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
import os
import pickle
import numpy as np

# Đọc dữ liệu đã xử lý 
df = pd.read_csv("data/processed/functions.csv")

# Khởi tạo TF-IDF vectorizer 
tfidf = TfidfVectorizer( 
 max_features=5000, # Giới hạn số lượng từ 
 ngram_range=(1, 3), # Sử dụng unigram, bigram và trigram  stop_words='english' # Loại bỏ stopwords 
) 
# Tạo ma trận TF-IDF 
tfidf_matrix = tfidf.fit_transform(df['processed_code']) 

# Chuyển ma trận thành DataFrame để dễ xem 
tfidf_df = pd.DataFrame( 
 tfidf_matrix.toarray(), 
 columns=tfidf.get_feature_names_out() 
) 

print(f"Shape of TF-IDF matrix: {tfidf_matrix.shape}")
print(tfidf_df.head()) 

# Tạo tất cả các thư mục cần thiết
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Lưu vectorizer để tái sử dụng 
try:
    with open("models/tfidf_vectorizer.pkl", "wb") as f: 
        pickle.dump(tfidf, f)
    print("Successfully saved vectorizer to models/tfidf_vectorizer.pkl")
except Exception as e:
    print(f"Error saving vectorizer: {e}")

# Lưu ma trận TF-IDF 
try:
    np.save("data/processed/tfidf_matrix.npy", tfidf_matrix.toarray())
    print("Successfully saved TF-IDF matrix to data/processed/tfidf_matrix.npy")
except Exception as e:
    print(f"Error saving TF-IDF matrix: {e}") 
