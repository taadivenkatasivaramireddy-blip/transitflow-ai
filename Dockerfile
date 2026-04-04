# ── Takshashila Transit OpenEnv — HuggingFace Spaces Dockerfile ──────────────
#
# Build:  docker build -t transit-openenv .
# Run:    docker run -p 7860:7860 transit-openenv
# HF:     Push to a HuggingFace Space (SDK: gradio)

FROM python:3.11-slim

# Metadata
LABEL maintainer="Ram <Takshashila University>"
LABEL description="Takshashila Transit OpenEnv — Real-world bus fleet RL environment"
LABEL version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HuggingFace Spaces requirement)
RUN useradd -m -u 1000 user
USER user

# Set up working directory
ENV HOME=/home/user
ENV PATH=$HOME/.local/bin:$PATH
WORKDIR $HOME/app

# Copy project
COPY --chown=user . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pydantic>=2.0 \
        gradio>=4.0 \
        pandas \
        plotly

# Expose port (HuggingFace Spaces uses 7860)
EXPOSE 7860

# Environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Launch Gradio app
CMD ["python", "app.py"]
