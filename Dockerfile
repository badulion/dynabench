FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /workspace

COPY scripts/ ./scripts/
COPY requirements.txt .
RUN ./scripts/install_requirements.sh
