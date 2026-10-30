# CS2 Maç Tahmin Motoru -- API + web arayüzü imajı (Hugging Face Spaces / Render vb.)
# Yerelde: docker build -t fragpredict . && docker run -p 7860:7860 fragpredict
FROM python:3.14-slim

# LightGBM'in Linux wheel'ı OpenMP'yi (libgomp) içinde taşımıyor
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces container'ı uid 1000 ile çalıştırır
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Predictor sadece bu iki parquet'i ve champion modeli okur (bkz. src/predict.py)
COPY --chown=user src/ src/
COPY --chown=user api/ api/
COPY --chown=user web/ web/
COPY --chown=user models/champion_model.pkl models/model_metadata.json models/
COPY --chown=user data/processed/maps_clean.parquet data/processed/matches_summary.parquet data/processed/

USER user
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
