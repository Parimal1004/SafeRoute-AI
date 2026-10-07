# Backend image (FastAPI + trained model). Frontend is deployed separately on Streamlit Community Cloud.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY . .
# Train at build time so the first request is fast (also retrains automatically if the model is missing)
RUN python -m ml.train_model
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
