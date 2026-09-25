FROM python:3.12-slim

# build-essential: some platforms (notably arm64) don't have prebuilt wheels
# for blis/thinc and fall back to compiling from source. Costs image size;
# drop it if you confirm your target platform always gets prebuilt wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_lg

COPY app ./app
COPY policy.yaml .

ENV POLICY_PATH=policy.yaml
ENV EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
