FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["gunicorn", "src.serve:create_app('models/model.joblib')", "-b", "0.0.0.0:8000", "--workers", "2"]
