FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=America/Chicago \
    PAPER_LOG_DIR=/app/logs

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-paper.txt ./
RUN pip install --no-cache-dir -r requirements-paper.txt

COPY mlb_kalshi/ ./mlb_kalshi/
COPY live_trading_engine/paper_trader.py ./live_trading_engine/paper_trader.py
COPY data/raw/scripts/download_live_kalshi_market_logs.py \
    ./data/raw/scripts/download_live_kalshi_market_logs.py
COPY models/market_reaction_model/ ./models/market_reaction_model/

RUN useradd --create-home --uid 10001 trader \
    && mkdir -p /app/logs \
    && chown -R trader:trader /app

USER trader

ENTRYPOINT ["python", "-u", "live_trading_engine/paper_trader.py"]
CMD ["--all-games"]
