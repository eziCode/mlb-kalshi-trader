FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PAPER_LOG_DIR=/app/paper_logs
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app/hit_reversion_strategy
WORKDIR /app/hit_reversion_strategy
CMD ["python", "scripts/backtest.py"]
