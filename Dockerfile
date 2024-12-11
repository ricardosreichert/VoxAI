# Usar uma imagem base do Python
FROM python:3.11-slim

# Definir o diretório de trabalho
WORKDIR /app

# Instalar Git LFS e outras dependências necessárias para o sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git-lfs \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Clonar o repositório XTTS com suporte a Git LFS
RUN git lfs install \
    && git clone https://huggingface.co/coqui/XTTS-v2

# Instalar pacotes Python necessários
RUN pip install --no-cache-dir TTS scipy

# Copiar o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instalar pacotes Python necessários
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o arquivo Python da aplicação para o diretório de trabalho
COPY main.py main.py

# Comando para rodar quando o contêiner iniciar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "65535"]
