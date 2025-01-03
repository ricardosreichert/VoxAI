ARG BASE=nvidia/cuda:11.8.0-base-ubuntu22.04
FROM ${BASE}

# System dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    ffmpeg \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-wheel \
    espeak-ng \
    libsndfile1-dev \
    wget \
    git-lfs \
    nano && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pip dependencies
RUN pip3 install --no-cache-dir llvmlite torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Clone the TTS repository
WORKDIR /root
RUN git clone https://github.com/coqui-ai/TTS.git /root/TTS

# Copy application files
COPY main.py /root
COPY xtts_handler.py /root
COPY audios/* /root/audios/

# Install TTS dependencies
WORKDIR /root/TTS
RUN make deps

# Clone the XTTS repository
WORKDIR /root
RUN git lfs install && git clone https://huggingface.co/coqui/XTTS-v2 /root/XTTS-v2

# Set write permissions for the working directory
RUN mkdir -p /root/audios && chmod -R 777 /root && chown -R 1000:1000 /root

# Install Python application dependencies
RUN pip install --no-cache-dir \
    librosa \
    python-dotenv \
    requests \
    websockets \
    uvicorn \
    fastapi \
    python-multipart \
    whisper \
    openai-whisper \
    "langchain==0.0.179"

# Define port argument and expose it
ARG PORT
ENV PORT=${PORT}
EXPOSE ${PORT}

# Start the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
