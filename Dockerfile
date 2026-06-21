FROM python:3.11-slim

# libgomp1  → required by TensorFlow (OpenMP runtime)
# libglib2.0-0 → required by some PIL/Pillow backends on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
ENV PYTHONPATH=/app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ────────────────────────────────────────────────────────────
COPY . .

# ── Streamlit port ──────────────────────────────────────────────────────────────
EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]