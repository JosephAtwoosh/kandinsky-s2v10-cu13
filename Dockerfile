# syntax=docker/dockerfile:1.7
#
# Kandinsky S2V 1.0 inference image (CUDA 13.0), speech-to-video: one reference frame
# plus one speech track -> a talking video at 25 fps.
#
# Why CUDA 13.0: upstream (kandinskylab/kandinsky-s2v) pins torch 2.13.0, torchvision and
# flash-attn-3 to the PyTorch cu130 index in its uv.lock. The host driver must be >= 580.
#
# Why Python 3.13 from uv: upstream requires Python >= 3.13 and Ubuntu 24.04 ships 3.12, so
# uv installs a standalone CPython 3.13 into /opt/python (headers included, which Triton
# needs to compile its launcher at runtime) and the runtime stage copies it over.
#
# Weights are NOT baked in (about 65 GB for Pro, 25 GB for Lite, partly from third-party repos).
# They go into the Hugging Face cache under /models/.hf: ks2v-download pro|lite.

ARG CUDA_VERSION=13.0.2
ARG UBUNTU_VERSION=24.04

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
# Pin the upstream source so the image is reproducible. Bump deliberately.
ARG KS2V_REPO=https://github.com/kandinskylab/kandinsky-s2v.git
ARG KS2V_REF=6d319f0ca20d69bfaf750d98d8d600ee4d39e8ca

ENV PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/ks2v/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv \
    && install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv --version

WORKDIR /opt/ks2v
RUN git init -q . \
    && git remote add origin "${KS2V_REPO}" \
    && git fetch -q --depth 1 origin "${KS2V_REF}" \
    && git checkout -q FETCH_HEAD \
    && test -f pyproject.toml && test -f uv.lock \
    && echo "kandinsky-s2v checked out at $(git rev-parse --short HEAD)"

# Upstream commits a uv.lock, so the environment is installed exactly as locked (--frozen),
# without the dev group. huggingface_hub[hf] adds the `hf` CLI for fetching weights.
RUN uv python install 3.13 \
    && uv sync --frozen --no-dev --python 3.13 \
    && uv pip install --python /opt/ks2v/.venv/bin/python "huggingface_hub[hf_xet]" \
    && /opt/ks2v/.venv/bin/python - <<'PY'
import sys, torch, diffusers, transformers
print(f"python {sys.version.split()[0]} | torch {torch.__version__} (CUDA build {torch.version.cuda})"
      f" | diffusers {diffusers.__version__} | transformers {transformers.__version__}")
PY

# VideoScore2 (TIGER-Lab/VideoScore2, Qwen2.5-VL-7B), the same scorer as the LTX-2.5 image, so takes
# from both models are judged alike. It pins torch 2.6 / transformers 4.53.2, so it gets its own venv
# on the system Python 3.12. Its 16.6 GB of weights stay outside the image (/models/videoscore2).
RUN uv venv /opt/vs2/venv --python /usr/bin/python3 \
    && uv pip install --python /opt/vs2/venv/bin/python \
         torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install --python /opt/vs2/venv/bin/python \
         transformers==4.53.2 qwen-vl-utils accelerate scipy opencv-python-headless numpy==2.2.6 "av==13.1.0" huggingface_hub \
    && /opt/vs2/venv/bin/python -c "import torch, transformers, qwen_vl_utils; print('vs2 torch', torch.__version__, 'transformers', transformers.__version__)"

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models/.hf \
    PATH=/opt/ks2v/.venv/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

# python3 (3.12) is for the VideoScore2 venv. build-essential: Triton compiles its CUDA launcher
# with `cc` the first time a Triton kernel runs (learned the hard way with the LTX-2.5 image).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        python3 \
        python3-dev \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        openssh-client \
        rsync \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/ks2v /opt/ks2v
COPY --from=builder /opt/vs2 /opt/vs2

COPY --chmod=0755 scripts/entrypoint.sh /usr/local/bin/ks2v-entrypoint
COPY --chmod=0755 scripts/doctor.sh /usr/local/bin/ks2v-doctor
COPY --chmod=0755 scripts/run-s2v.sh /usr/local/bin/ks2v-run
COPY --chmod=0755 scripts/download-models.sh /usr/local/bin/ks2v-download
COPY --chmod=0755 scripts/vs2_score.py /usr/local/bin/ks2v-vs2-score

RUN mkdir -p /models /output /workspace
VOLUME ["/models", "/output"]

WORKDIR /opt/ks2v
ENTRYPOINT ["/usr/local/bin/ks2v-entrypoint"]
CMD ["doctor"]
