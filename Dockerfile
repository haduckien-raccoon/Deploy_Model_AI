FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --trusted-host download.pytorch.org

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]