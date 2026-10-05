FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/home/prism/.cache/huggingface
# Install CPU-only torch first to avoid unnecessary CUDA dependencies on amd64.
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels \
    && useradd --create-home --uid 10001 prism \
    && mkdir -p /app /home/prism/.cache/huggingface \
    && chown -R prism:prism /app /home/prism/.cache
USER prism
WORKDIR /app
RUN prism init --preset openrouter
EXPOSE 8080
HEALTHCHECK --start-period=300s --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"
ENTRYPOINT ["prism"]
CMD ["start", "--profile", "/app/profiles/prism-balanced.json", "--host", "0.0.0.0", "--port", "8080"]
