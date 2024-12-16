ARG BASE=nvidia/cuda:11.8.0-base-ubuntu22.04
FROM ${BASE}

# System dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
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

# Install Dependencies:
RUN pip install librosa requests websockets uvicorn fastapi python-multipart whisper openai-whisper "langchain==0.0.179"

EXPOSE 7777

# Start an interactive shell
#CMD ["tail", "-f", "/dev/null"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7777"] 