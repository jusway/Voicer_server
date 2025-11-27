# config.py
import os
from dotenv import load_dotenv
import dashscope

# load_dotenv() 会自动查找同目录下的 .env 文件
load_dotenv()

# --- 阿里百炼 ---
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

if not dashscope.api_key:
    print("CRITICAL ERROR: DASHSCOPE_API_KEY not found or is empty. Please check your .env file.")
    raise ValueError("DASHSCOPE_API_KEY not found.")

# --- New API 中转站 ---
NEW_API_BASE_URL = os.getenv("NEW_API_BASE_URL")
NEW_API_KEY = os.getenv("NEW_API_KEY")

# 检查 New API 变量 (可选, 但推荐)
if not all([NEW_API_BASE_URL, NEW_API_KEY]):
    print("WARNING: NEW_API_* variables not fully set. New API LLM will fail if used.")
    


print("Config: .env file loaded and all configs set.")