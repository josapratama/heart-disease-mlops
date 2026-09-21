FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model
COPY serving/app.py .
COPY serving/model ./model

ENV MODEL_DIR=/app/model/heart_disease_model/1
ENV PORT=8080

EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
