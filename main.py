import asyncio
import websockets
import whisper
import tempfile
import wave
import os
import tempfile
import requests
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import warnings
from fastapi.responses import HTMLResponse

# Suprimir o aviso específico de FP16 no Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Suprime o aviso específico sobre `weights_only` no Whisper
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)

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
model = whisper.load_model("tiny").to("cpu")


# URL do endpoint do modelo LLaMA 3.2 no contêiner local
LLAMA_ENDPOINT = "http://localhost:11434/api/generate"

@app.get("/")
async def get():
    return HTMLResponse("<h1>WebSocket Audio Transcription Server</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket conectado")

    audio_chunks = []
    try:
        while True:
            # Recebe o áudio como um blob binário
            audio_chunk = await websocket.receive_bytes()
            audio_chunks.append(audio_chunk)

            # Temporariamente salva os dados recebidos para processar em tempo real
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_chunk)
                temp_file_path = temp_file.name

            # Processa o áudio com Whisper
            transcription = model.transcribe(temp_file_path)

            # Exibe a transcrição no console
            print("Transcrição:", transcription["text"])

            # Envia a transcrição ao modelo LLaMA 3.2 e recebe a resposta
            response = process_with_llama(transcription["text"])
            print("Resposta do LLaMA:", response)

            # Envia a transcrição de volta ao cliente
            await websocket.send_text(response)

            # Remove o arquivo temporário
            os.remove(temp_file_path)

    except websockets.exceptions.ConnectionClosedError:
        print("Conexão WebSocket fechada")

    except Exception as e:
        print(f"Erro durante a transcrição: {e}")

    finally:
        # Fecha a conexão WebSocket apenas se ainda estiver aberta
        if not connection_closed:
            await websocket.close()
        print("WebSocket desconectado")


def process_with_llama(transcription: str) -> str:
    """Envia o texto transcrito para o modelo LLaMA 3.2 e retorna a resposta."""
    try:
        # Formata a requisição conforme o exemplo especificado
        payload = {
            "prompt": transcription,
            "stream": False,
            "model": "llama3.2"
        }

        headers = {
            "Content-Type": "application/json"
        }

        # Log para inspecionar o payload antes de enviá-lo
        print("Payload enviado para LLaMA:", payload)

        # Realiza a requisição HTTP POST para o modelo LLaMA
        response = requests.post(LLAMA_ENDPOINT, json=payload, headers=headers)

        print("Response: ", response)

        response.raise_for_status()

        # Extrai o texto da resposta
        llama_response = response.json().get("response", "Erro na resposta do LLaMA")
        return llama_response

    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar com o LLaMA: {e}")
        return "Erro ao processar a resposta do modelo LLaMA."
