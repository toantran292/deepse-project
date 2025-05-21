# Sử dụng GitHub API 
import requests 
import base64 
import os

# Thiết lập thông tin API 
github_token = "aaaa" # Tạo token từ GitHub settings headers = { 
headers = { 
 "Authorization": f"token {github_token}", 
 "Accept": "application/vnd.github.v3+json" 
} 

# Hàm lấy nội dung file từ repository 
def get_file_content(owner, repo, path, branch="main"): 
 url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}" 
 response = requests.get(url, headers=headers) 
 if response.status_code == 200: 
  content = response.json() 
  
  # Kiểm tra nếu là file 
  if "type" in content and content["type"] == "file":  
    return base64.b64decode(content["content"]).decode("utf-8")
  
  return None 

# Lấy danh sách các file Python trong một repository 
def get_python_files(owner, repo, path="", branch="main"):  
 url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}" 
 response = requests.get(url, headers=headers) 
 files = [] 
  
 if response.status_code == 200: 
  contents = response.json()
  for item in contents: 
    if item["type"] == "file" and item["name"].endswith(".py"):
      files.append(item["path"]) 
    elif item["type"] == "dir": 
    # Đệ quy cho thư mục con 
      files.extend(get_python_files(owner, repo, item["path"], branch)) 
  
  return files 

owner = "tensorflow" 
repo = "models" 
python_files = get_python_files(owner, repo, path="official/legacy/bert",  branch="master")

# Lưu các file vào thư mục local 
os.makedirs("data/raw", exist_ok=True)
for i, file_path in enumerate(python_files[:10]): # Lấy 10 file đầu  tiên 
 content = get_file_content(owner, repo, file_path, branch="master")
 if content: 
  local_path = f"data/raw/file_{i}.py" 
  with open(local_path, "w", encoding="utf-8") as f:  f.write(content) 
  print(f"Saved to {local_path}")
