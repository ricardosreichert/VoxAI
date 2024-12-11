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
import warnings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from xtts_handler import XTTSHandler

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
# Definir cpu ou cuda
whisper_model = whisper.load_model("tiny").to("cuda")

# URL do endpoint do modelo LLaMA 3.2 no contêiner local
LLAMA_ENDPOINT = "http://localhost:11434/api/generate"

class LLAMAEndpointLLM(LLM):
    """Custom LLM class to interact with LLaMA via HTTP endpoint."""

    endpoint_url: str

    def _call(self, prompt: str, stop=None) -> str:
        payload = {
            "prompt": prompt,
            "stream": False,
            "model": "llama3.2"
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

# Cria uma instância do manipulador XTTS
xtts_handler = XTTSHandler()

@app.get("/")
async def get():
    return HTMLResponse("<h1>WebSocket Audio Transcription Server</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket conectado")

    try:
        while True:
            # Recebe o áudio como um blob binário
            audio_chunk = await websocket.receive_bytes()

            # Temporariamente salva os dados recebidos para processar em tempo real
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_chunk)
                temp_file_path = temp_file.name

            # Processa o áudio com Whisper
            transcription = whisper_model.transcribe(temp_file_path)

            # Exibe a transcrição no console
            print("Transcrição:", transcription["text"])

            # Envia um objeto JSON com a transcrição
            await websocket.send_json({
                "transcription": transcription["text"]
            })

            # Envia a transcrição ao modelo LLaMA e recebe a resposta
            llama_response = process_with_llama(transcription["text"])
            print("Resposta do LLaMA:", llama_response)

            # Envia um objeto JSON com a transcrição e a resposta do LLaMA de volta ao cliente
            await websocket.send_json({
                "transcription": transcription["text"],
                "llama": llama_response
            })

            # Define o nome da voz (ajuste conforme necessário)
            voice_name = "man"  # man ou "woman"

            # Sintetiza a fala com XTTS
            audio_bytes = xtts_handler.synthesize(llama_response, voice_name)

            # Gera um nome de arquivo para salvar o áudio sintetizado
            filename = f"audios/generated_{int(time.time())}.wav"

            # Salva o arquivo de áudio na pasta 'audios/'
            with open(filename, "wb") as f:
                f.write(audio_bytes)

            # Envia o áudio pelo socket
            await websocket.send_bytes(audio_bytes)

            # Remove o arquivo temporário
            os.remove(temp_file_path)

    except websockets.exceptions.ConnectionClosedError:
        print("Conexão WebSocket fechada")

    except Exception as e:
        print(f"Erro durante a transcrição: {e}")

    finally:
        # Fecha a conexão WebSocket
        await websocket.close()
        print("WebSocket desconectado")

def process_with_llama(transcription: str) -> str:
    """Usa LangChain para processar a transcrição com o modelo LLaMA."""
    try:
        # Formata o prompt manualmente
        prompt_text = prompt.format(input=transcription)

        response = llm(prompt_text)
        return response.strip()
    except Exception as e:
        print(f"Erro ao processar com LLaMA: {e}")
        return "Erro ao processar a resposta do modelo LLaMA."

