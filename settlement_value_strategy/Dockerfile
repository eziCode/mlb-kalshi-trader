FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PAPER_LOG_DIR=/app/paper_logs \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

WORKDIR /app
COPY settlement_value_strategy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt 'tzdata>=2025.2'
COPY settlement_value_strategy /app/settlement_value_strategy
COPY data/settlement_value /app/data/settlement_value

ENTRYPOINT ["python", "-m", "settlement_value_strategy.backtest"]
