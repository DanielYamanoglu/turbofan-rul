FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY models ./models
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TURBOFAN_MODEL_DIR=/app/models
# Numeric non-root user, so Kubernetes can enforce runAsNonRoot.
RUN useradd --uid 10001 --no-create-home app
USER 10001
EXPOSE 8000
CMD ["uvicorn", "turbofan_rul.api:app", "--host", "0.0.0.0", "--port", "8000"]
