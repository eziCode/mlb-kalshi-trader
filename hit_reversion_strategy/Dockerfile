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
COPY hit_reversion_strategy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt 'tzdata>=2025.2'
COPY hit_reversion_strategy /app/hit_reversion_strategy
COPY data/shared /app/data/shared
WORKDIR /app/hit_reversion_strategy
CMD ["python", "scripts/backtest.py"]
