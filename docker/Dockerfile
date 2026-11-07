# One image for every role (coordinator / active / passive / tools).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt
COPY . .
EXPOSE 50050 50051 8080 8081 8082
ENTRYPOINT ["python", "-m", "vfl.cli"]
