FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PAPER_LOG_DIR=/app/paper_logs

WORKDIR /app

# Install both dependency sets. The portable lock supplies reproducible
# versions for their shared stack; scikit-learn is retained for mispricing.
COPY settlement_value_strategy/requirements.txt /tmp/mispricing-requirements.txt
COPY hit_reversion_strategy/requirements.txt /tmp/trade-tape-requirements.txt
RUN pip install --no-cache-dir \
      -r /tmp/mispricing-requirements.txt \
      -r /tmp/trade-tape-requirements.txt

COPY settlement_value_strategy /app/settlement_value_strategy
COPY hit_reversion_strategy /app/hit_reversion_strategy
COPY data /app/data
COPY setup_data.py /app/setup_data.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

ENTRYPOINT ["/bin/sh", "/app/docker-entrypoint.sh"]
CMD ["help"]
