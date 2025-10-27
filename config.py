import os
from dotenv import load_dotenv
import dashscope

# load_dotenv() 会自动查找同目录下的 .env 文件
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 检查一下是否加载成功
if not dashscope.api_key:
    print("CRITICAL ERROR: DASHSCOPE_API_KEY not found or is empty. Please check your .env file.")
    raise ValueError("DASHSCOPE_API_KEY not found.")
print("Config: .env file loaded.")

