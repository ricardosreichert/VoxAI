import asyncio
import websockets
import whisper
import tempfile
import wave
import os
import time
import requests
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
import torch
import warnings_filters

# Carrega as variáveis do arquivo .env
load_dotenv()

# Variáveis de ambiente
PORT = os.getenv("PORT")
LLAMA_MODEL = os.getenv("LLAMA_MODEL")
LLAMA_ENDPOINT = os.getenv("LLAMA_ENDPOINT")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # Padrão para 'cuda' se não especificado
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")  # Modelo padrão: 'base'
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
VOICE_NAME = os.getenv("VOICE_NAME", "man")  # man ou "woman"

# Configuração do logger
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,  # Ajusta o nível de log com base no DEBUG_MODE
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Verifica a disponibilidade de GPU para ajustar Whisper e XTTS
gpu_available = torch.cuda.is_available()
logger.info(f"GPU disponível: {gpu_available}")
if not gpu_available:
    WHISPER_DEVICE = "cpu"  # Força o uso de CPU para o Whisper
    logger.warning("GPU não disponível. Whisper será executado no modo CPU.")

logger.debug(f"Running API on port {PORT}")
logger.debug(f"Using LLaMA model: {LLAMA_MODEL}")
logger.debug(f"LLaMA endpoint: {LLAMA_ENDPOINT}")
logger.debug(f"Whisper model '{WHISPER_MODEL_NAME}' initialized on device: {WHISPER_DEVICE}")

app = FastAPI()

# Permitir todas as origens (você pode restringir para um domínio específico)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o modelo Whisper
whisper_model = whisper.load_model(WHISPER_MODEL_NAME).to(WHISPER_DEVICE)

if gpu_available:
    from xtts_handler import XTTSHandler
    # Cria uma instância do manipulador XTTS
    xtts_handler = XTTSHandler()
else:
    logger.warning("GPU não disponível. Módulo XTTS será desativado.")
    xtts_handler = None

class LLAMAEndpointLLM(LLM):
    """Custom LLM class to interact with LLaMA via HTTP endpoint."""

    endpoint_url: str

    def _call(self, prompt: str, stop=None) -> str:
        payload = {
            "prompt": prompt,
            "stream": False,
            "model": LLAMA_MODEL
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(self.endpoint_url, json=payload, headers=headers)
        response.raise_for_status()
        llama_response = response.json().get("response", "Erro na resposta do LLaMA")
        return llama_response

    @property
    def _identifying_params(self):
        return {"endpoint_url": self.endpoint_url}

    @property
    def _llm_type(self) -> str:
        return "llama_endpoint_llm"

# Instancia o LLM personalizado
llm = LLAMAEndpointLLM(endpoint_url=LLAMA_ENDPOINT)

# Define um template de prompt (opcional)
prompt_template = """Você é um assistente feminino, útil, prestativo, que responde sempre em portugues de forma curta e objetiva.

Usuário: {input}

Assistente:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["input"])

@app.get("/")
async def get():
    return HTMLResponse("<h1>WebSocket Audio Transcription Server</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.debug("Tentando aceitar conexão WebSocket")
    await websocket.accept()
    logger.info("WebSocket conectado")

    try:
        while True:
            logger.debug("Aguardando dados do cliente via WebSocket")
            # Recebe o áudio como um blob binário
            audio_chunk = await websocket.receive_bytes()
            logger.debug(f"Recebido {len(audio_chunk)} bytes de áudio")

            # Temporariamente salva os dados recebidos para processar em tempo real
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_chunk)
                temp_file_path = temp_file.name
            logger.debug(f"Áudio salvo temporariamente em {temp_file_path}")

            # Processa o áudio com Whisper
            transcription = whisper_model.transcribe(temp_file_path)
            logger.info(f"Transcrição gerada: {transcription['text']}")

            # Envia um objeto JSON com a transcrição
            await websocket.send_json({
                "transcription": transcription["text"]
            })
            logger.debug("Transcrição enviada ao cliente")

            # Envia a transcrição ao modelo LLaMA e recebe a resposta
            llama_response = process_with_llama(transcription["text"])
            logger.info(f"Resposta do LLaMA: {llama_response}")

            # Envia um objeto JSON com a transcrição e a resposta do LLaMA de volta ao cliente
            await websocket.send_json({
                "transcription": transcription["text"],
                "llama": llama_response
            })
            logger.debug("Resposta do LLaMA enviada ao cliente")

            if xtts_handler:
                # Sintetiza a fala com XTTS
                audio_bytes = xtts_handler.synthesize(llama_response, VOICE_NAME)
                logger.debug("Áudio sintetizado gerado com sucesso")

                # Gera um nome de arquivo para salvar o áudio sintetizado
                filename = f"audios/generated_{int(time.time())}.wav"

                # Salva o arquivo de áudio na pasta 'audios/'
                with open(filename, "wb") as f:
                    f.write(audio_bytes)
                logger.debug(f"Áudio sintetizado salvo em {filename}")

                # Envia o áudio pelo socket
                await websocket.send_bytes(audio_bytes)
                logger.debug("Áudio sintetizado enviado ao cliente")

            # Remove o arquivo temporário
            os.remove(temp_file_path)
            logger.debug(f"Arquivo temporário {temp_file_path} removido")

    except websockets.exceptions.ConnectionClosedError:
        logger.warning("Conexão WebSocket fechada pelo cliente")

    except Exception as e:
        logger.error(f"Erro durante a execução: {e}")

    finally:
        # Certifica-se de que o WebSocket não está fechado antes de tentar fechá-lo
        if not websocket.client_state.name == "DISCONNECTED":
            await websocket.close()
            logger.info("WebSocket desconectado")

def process_with_llama(transcription: str) -> str:
    """Usa LangChain para processar a transcrição com o modelo LLaMA."""
    try:
        # Formata o prompt manualmente
        prompt_text = prompt.format(input=transcription)

        response = llm(prompt_text)
        return response.strip()
    except Exception as e:
        logger.error(f"Erro ao processar com LLaMA: {e}")
        return "Erro ao processar a resposta do modelo LLaMA."
