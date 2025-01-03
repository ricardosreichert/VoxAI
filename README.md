# Projeto: Servidor de Transcrição de Áudio e Integração com LLaMA

Este projeto é uma API construída com FastAPI que processa áudio recebido via WebSocket. Ele transcreve o áudio usando Whisper, processa o texto com o modelo LLaMA e sintetiza uma resposta em áudio utilizando XTTS. Tudo é configurado e executado em contêineres Docker para facilitar a instalação e a execução.

## Funcionalidades

- **Transcrição de áudio:** Recebe áudio via WebSocket e transcreve utilizando Whisper.
- **Processamento de texto:** Processa a transcrição com o modelo LLaMA para gerar respostas inteligentes.
- **Síntese de áudio:** Converte a resposta gerada em áudio usando XTTS.
- **Execução em Docker:** Todos os componentes são configurados para rodar em contêineres Docker.

## Requisitos

- [Docker](https://www.docker.com/) versão 20.10 ou superior
- [Docker Compose](https://docs.docker.com/compose/) versão 1.29 ou superior

## Como instalar e executar

### 1. Configurar o arquivo `.env`

Crie um arquivo `.env` na raiz do projeto com base no exemplo `.env.example`. Preencha as variáveis necessárias:

```env
PORT=8000
LLAMA_MODEL=nome_do_modelo_llama
LLAMA_ENDPOINT=http://localhost:11434
WHISPER_MODEL=base
WHISPER_DEVICE=cuda
DEBUG_MODE=true
VOICE_NAME=man
```

### 2. Construir e iniciar os contêineres

Na raiz do projeto, execute o seguinte comando:

```bash
docker-compose up --build
```

Isso irá:

- Construir e iniciar o serviço de API.
- Iniciar o contêiner do modelo LLaMA.

### 3. Usar uma imagem do Docker Hub

Caso prefira, você pode usar a imagem disponível no Docker Hub em vez de construir localmente. Execute:

```bash
docker pull ricardoreichert/NOMEDAIMAGEM:tagname
```

Substitua `NOMEDAIMAGEM` e `tagname` pelos valores correspondentes da imagem no repositório.

Se optar por usar a imagem do Docker Hub, será necessário ajustar o arquivo `docker-compose.yml` para referenciar a imagem ao invés de construir localmente. Por exemplo:

```yaml
services:
  api:
    image: ricardoreichert/NOMEDAIMAGEM:tagname
    container_name: api_container
    ports:
      - '${PORT}:${PORT}'
    environment:
      - LLAMA_ENDPOINT=$LLAMA_ENDPOINT
      - PORT=$PORT
      - WHISPER_MODEL=$WHISPER_MODEL
      - WHISPER_DEVICE=$WHISPER_DEVICE
    env_file:
      - .env
    networks:
      - app_network
```

### 4. Testar o servidor

### Formato de Request e Response da API

#### Endpoint WebSocket `/ws`

- **Request:**
  Envie um arquivo de áudio no formato binário pelo WebSocket. O áudio deve estar no formato WAV.

- **Response:**
  O servidor retorna uma resposta JSON contendo a transcrição e a resposta gerada pelo modelo LLaMA, além de enviar o áudio sintetizado:

  ```json
  {
    "transcription": "Texto transcrito do áudio",
    "llama": "Resposta gerada pelo modelo LLaMA"
  }
  ```

  Em seguida, o servidor também envia o áudio sintetizado no formato WAV como bytes pelo WebSocket.

Após iniciar os contêineres:

- A API estará disponível em `http://localhost:8000`.
- A conexão WebSocket para envio de áudio pode ser feita em `ws://localhost:8000/ws`.

### 5. Enviar áudio via WebSocket

Conecte-se ao endpoint WebSocket para enviar arquivos de áudio e receber respostas transcritas e sintetizadas.

### 6. Logs e depuração

Os logs do servidor são configurados para exibir mensagens no terminal. Ative o modo de depuração (`DEBUG_MODE=true`) no arquivo `.env` para obter informações detalhadas.

## Estrutura do Projeto

- `main.py`: Contém a API e o processamento principal.
- `xtts_handler.py`: Módulo responsável por gerenciar a síntese de áudio usando XTTS.
- `Dockerfile`: Define como a imagem Docker do serviço é construída.
- `docker-compose.yml`: Configuração para orquestrar os contêineres do projeto.

## Observações

- É obrigatório o uso de GPU para o funcionamento completo da API, incluindo a transcrição com Whisper e o processamento com LLaMA.

- Certifique-se de que o modelo LLaMA esteja configurado corretamente no contêiner `ollama-ai`.
- Mais informações sobre o funcionamento do contêiner Ollama estão disponíveis em [Ollama Docker Hub](https://hub.docker.com/r/ollama/ollama).
- O Whisper requer uma GPU para funcionar com melhor desempenho. Certifique-se de que o driver NVIDIA esteja instalado e funcionando corretamente.

## Problemas Comuns

- **Conexão recusada no endpoint LLaMA:** Verifique se o contêiner `ollama-ai` está ativo e acessível na porta correta.
- **Erro na síntese de áudio:** Certifique-se de que os arquivos de referência de voz estejam disponíveis na pasta `audios/`.
