# MOG engine: API server and render worker share this image.
# Ubuntu 24.04: FFmpeg 6.1 + Python 3.12, the same versions the engine is tested with.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY engine/pyproject.toml engine/pyproject.toml
COPY engine/mog engine/mog
RUN pip install --no-cache-dir "./engine[server]"

# Optional ML beat tracker (PyTorch CPU build, ~700 MB). Without it the librosa tracker is used.
#   docker compose build --build-arg INSTALL_BEATS=1
ARG INSTALL_BEATS=0
RUN if [ "$INSTALL_BEATS" = "1" ]; then \
      pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir "beat_this @ https://github.com/CPJKU/beat_this/archive/main.zip"; \
    fi

COPY presets presets
COPY assets assets

ENV MOG_ROOT=/app \
    MOG_DATA=/data \
    MOG_CACHE=/data/cache \
    PYTHONUNBUFFERED=1

# ubuntu:24.04 ships a uid-1000 "ubuntu" user: run as that, never as root
RUN mkdir -p /data && chown ubuntu /data
USER ubuntu

EXPOSE 8000
CMD ["uvicorn", "mog.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
