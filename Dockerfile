# Image for Hugging Face Spaces (Docker SDK). Also runs anywhere Docker does:
#   docker build -t skinsense . && docker run -p 8501:8501 skinsense
FROM python:3.12-slim

# Spaces run containers as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# Generate the sample PDF and download all models at build time, so visitors don't wait.
RUN python scripts/make_sample_pdf.py && python scripts/preload_models.py

EXPOSE 8501
# XSRF protection must be off for file uploads to work inside the Spaces iframe.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false"]
