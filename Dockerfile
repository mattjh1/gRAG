FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
WORKDIR /app

COPY src/requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --upgrade chainlit

COPY src .

ENV PYTHONPATH=/app

EXPOSE 8000
