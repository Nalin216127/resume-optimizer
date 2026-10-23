# syntax=docker/dockerfile:1
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --uid 10001 --create-home app
WORKDIR /app
COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt
# The UI only imports the samples module from the package in API mode.
COPY resume_optimizer ./resume_optimizer
COPY ui ./ui
USER 10001
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
