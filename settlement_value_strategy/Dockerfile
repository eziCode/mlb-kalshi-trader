FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PAPER_LOG_DIR=/app/paper_logs

WORKDIR /app
COPY settlement_value_strategy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY settlement_value_strategy /app/settlement_value_strategy

ENTRYPOINT ["python", "-m", "settlement_value_strategy.backtest"]

