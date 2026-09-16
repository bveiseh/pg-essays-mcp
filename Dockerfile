FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PUBLIC_HOST=pg-essays-mcp.fly.dev

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY indexer.py search.py server.py usage.py entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh
RUN DATA_DIR=/seed-data python /app/indexer.py

ENV DATA_DIR=/data

EXPOSE 8080
CMD ["/app/entrypoint.sh"]
