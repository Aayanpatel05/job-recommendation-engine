import requests

url = "http://127.0.0.1:8000/recommend"
resume_path = "C:/Users/patel/Downloads/resume.pdf"

with open(resume_path, "rb") as f:
    files = {"file": ("resume.pdf", f, "application/pdf")}  # change "resume" -> "file"
    response = requests.post(url, files=files)

print(response.status_code)
print(response.text)