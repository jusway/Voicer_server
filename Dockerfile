# 基于 Linux 的 Python 镜像（适用于在 Linux 服务器上运行）
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 可选：如需处理音视频/长音频，推荐安装 ffmpeg 和 sndfile
# 如果不需要，可注释掉下面一行
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖，再安装，提升构建缓存利用率
COPY requirements.txt .
RUN pip install --timeout=900 --no-cache-dir -r requirements.txt

# 拷贝项目源代码
COPY . .

# Gradio 默认端口
EXPOSE 7860

# 直接运行 app.py（Gradio 已在 app.py 中设置 0.0.0.0/7860）
CMD ["python", "app.py"]