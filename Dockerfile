ARG BASE=nvidia/cuda:11.8.0-base-ubuntu22.04
FROM ${BASE}

# System dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends nano gcc g++ make python3 python3-dev python3-pip python3-venv python3-wheel espeak-ng libsndfile1-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install llvmlite --ignore-installed

# Install Dependencies:
RUN pip3 install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN rm -rf /root/.cache/pip

# Clone the TTS repository
WORKDIR /root
RUN git clone https://github.com/coqui-ai/TTS.git /root/TTS

# Install TTS
COPY main.py /root
COPY xtts_handler.py /root
COPY audios/* /root/audios/

WORKDIR /root/TTS
RUN make deps

# Clone the XTTS repository
WORKDIR /root
RUN git lfs install && git clone https://huggingface.co/coqui/XTTS-v2 /root/XTTS-v2

# Set write permissions for the working directory
RUN mkdir -p /root/audios && chmod -R 777 /root
RUN chown -R 1000:1000 /root

# Install Dependencies:
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

# Define o argumento PORT vindo do docker-compose
ARG PORT

# Exporta a porta definida pelo argumento
ENV PORT=${PORT}
EXPOSE ${PORT}


# Define a permissão para criação e escrita de arquivos
RUN chmod -R 777 /root

# Start an interactive shell
#CMD ["tail", "-f", "/dev/null"]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]