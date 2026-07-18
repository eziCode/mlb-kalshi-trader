FROM python:3.12-slim

ARG APPROACH
ENV APPROACH=${APPROACH} \
    PAPER_MODE=replay \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements-paper.txt ./requirements-paper.txt
RUN pip install --no-cache-dir -r requirements-paper.txt

COPY approaches/ ./approaches/
COPY mlb_kalshi/ ./mlb_kalshi/
COPY models/ ./models/
COPY backtesting/ ./backtesting/
COPY live_trading_engine/ ./live_trading_engine/
COPY data/processed/ ./data/processed/
COPY studies/ ./studies/

RUN useradd --create-home --uid 10001 trader && chown -R trader:trader /app
USER trader

ENTRYPOINT ["sh", "-c", "python -u -m approaches.${APPROACH}.paper_trader \"$@\"", "--"]
