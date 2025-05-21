import ast 
import re 
def preprocess_python_code(code): 
 # Loại bỏ comments 
 code = re.sub(r'#.*', '', code) 
 code = re.sub(r'"""[\s\S]*?"""', '', code) 
 code = re.sub(r"'''[\s\S]*?'''", '', code) 
  
 # Chuẩn hóa khoảng trắng 
 code = re.sub(r'\s+', ' ', code) 
  
 return code.strip() 
def extract_functions(code): 
 try: 
  tree = ast.parse(code) 
  functions = [] 
  
  for node in ast.walk(tree): 
    if isinstance(node, ast.FunctionDef): 
      func_code = ast.get_source_segment(code, node)
      functions.append({ 
        'name': node.name, 
        'code': func_code, 
        'processed_code': preprocess_python_code(func_code) 
      }) 
  
  return functions 
 except SyntaxError: 
  return [] 
 
# Ví dụ sử dụng 
import os 
processed_data = [] 
for i in range(10): # Cho 10 file đã tải 
 file_path = f"data/raw/file_{i}.py"
 if os.path.exists(file_path): 
  with open(file_path, "r", encoding="utf-8") as f:  code = f.read() 

  # Tiền xử lý và trích xuất hàm 
  functions = extract_functions(code) 
  processed_data.extend(functions) 

print(f"Extracted {len(processed_data)} functions") 
# Lưu dữ liệu đã xử lý 

import pandas as pd 
df = pd.DataFrame(processed_data) 
df.to_csv("data/processed/functions.csv", index=False) 
