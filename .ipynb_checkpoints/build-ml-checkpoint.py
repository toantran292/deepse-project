import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
# Đọc dữ liệu vector 
tfidf_vectors = np.load("data/processed/tfidf_matrix.npy")
w2v_vectors = np.load("data/processed/w2v_vectors.npy") 
# Giả định: Gán nhãn cho mỗi hàm (ví dụ: phân loại theo chức năng) 
# Trong thực tế, bạn cần có dữ liệu đã được gán nhãn 
# Ở đây, chúng ta tạo nhãn giả cho mục đích demo 
df = pd.read_csv("data/processed/functions.csv") 
# Ví dụ: Phân loại hàm theo tên 
# 0: hàm bắt đầu bằng "get_" hoặc "fetch_" 
# 1: hàm bắt đầu bằng "create_" hoặc "build_" 
# 2: các hàm còn lại 
def assign_label(func_name): 
 if func_name.startswith(('get_', 'fetch_')): 
  return 0 
 elif func_name.startswith(('create_', 'build_')): 
  return 1 
 else: 
  return 2 
df['label'] = df['name'].apply(assign_label) 
# Chia dữ liệu 
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(  tfidf_vectors, df['label'], test_size=0.3, random_state=42) 
X_train_w2v, X_test_w2v, _, _ = train_test_split(w2v_vectors, df['label'], test_size=0.3, random_state=42) 

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, accuracy_score 

print("\n\nMô hình SVM:")
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_tfidf, y_train) 
# Dự đoán 
y_pred = svm_model.predict(X_test_tfidf) 
# Đánh giá 
print("SVM với TF-IDF:") 
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, zero_division=0)) 
# Lưu mô hình 
import pickle 
with open("models/svm_tfidf.pkl", "wb") as f: 
  pickle.dump(svm_model, f) 


from sklearn.ensemble import RandomForestClassifier 
print("\n\nMô hình Random Forest:")
rf_model = RandomForestClassifier( 
 n_estimators=100, 
 max_depth=10, 
 random_state=42 
) 
rf_model.fit(X_train_w2v, y_train) 
# Dự đoán 
y_pred = rf_model.predict(X_test_w2v) 
# Đánh giá 
print("\nRandom Forest với Word2Vec:") 
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, zero_division=0)) 
# Lưu mô hình 
with open("models/rf_w2v.pkl", "wb") as f: 
 pickle.dump(rf_model, f) 

print("\n\nHuấn luyện Random Forest với TF-IDF để so sánh :")
rf_tfidf = RandomForestClassifier( 
 n_estimators=100,
 max_depth=10, 
 random_state=42 
) 
rf_tfidf.fit(X_train_tfidf, y_train) 
y_pred_tfidf = rf_tfidf.predict(X_test_tfidf) 
# Đánh giá 
print("\nRandom Forest với TF-IDF:") 
print(f"Accuracy: {accuracy_score(y_test, y_pred_tfidf):.4f}")
print(classification_report(y_test, y_pred_tfidf, zero_division=0)) 
# Kết luận 
print("\nSo sánh biểu diễn vector:") 
print(f"SVM + TF-IDF: {accuracy_score(y_test, svm_model.predict(X_test_tfidf)):.4f}") 
print(f"RF + Word2Vec: {accuracy_score(y_test,  rf_model.predict(X_test_w2v)):.4f}") 
print(f"RF + TF-IDF: {accuracy_score(y_test, y_pred_tfidf):.4f}")
