# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --uid 10001 --create-home app
COPY --from=builder /install /usr/local
WORKDIR /app
COPY resume_optimizer ./resume_optimizer
RUN mkdir -p /app/data && chown app:app /app/data
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "resume_optimizer.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
