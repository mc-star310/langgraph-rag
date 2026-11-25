FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Ollama 以独立服务运行（见 docker-compose.yml），此处仅启动 Web 服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
