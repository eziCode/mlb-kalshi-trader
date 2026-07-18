FROM python:3.12-slim
ENV PAPER_MODE=live PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-paper.txt ./
RUN pip install --no-cache-dir -r requirements-paper.txt
COPY . .
RUN useradd --create-home --uid 10001 trader && mkdir -p /app/logs && chown -R trader:trader /app
USER trader
ENTRYPOINT ["python", "-u", "-m", "approaches.trade_tape_hybrid.paper_trader"]
CMD ["--all-games"]
