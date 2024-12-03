from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from scipy.io.wavfile import write
import io
import os

class XtTSHandler:
    def __init__(self, model_path="./XTTS-v2/", config_path="./XTTS-v2/config.json", audios_dir="./audios"):
        self.config = XttsConfig()
        self.config.load_json(config_path)
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=model_path)
        self.model.cuda()  # Mover para a GPU se disponível
        self.audios_dir = audios_dir

    def synthesize(self, text, voice_name):
        """
        Sintetiza a fala usando o modelo XTTS e retorna o áudio como bytes.

        Args:
            text (str): Texto a ser sintetizado.
            voice_name (str): Nome da voz desejada (ex: "man", "woman").

        Returns:
            bytes: Áudio gerado como bytes.
        """
        reference_audio_path = os.path.join(self.audios_dir, f"{voice_name}.wav")

        outputs = self.model.synthesize(
            text,
            self.config,
            speaker_wav=[reference_audio_path],  # Usar o caminho do áudio de referência
            gpt_cond_len=3,
            language="pt",
        )

        # Salvar o áudio em um buffer de memória
        audio_buffer = io.BytesIO()
        write(audio_buffer, 24000, outputs['wav'])
        audio_buffer.seek(0)  # Reposicionar o cursor para o início do buffer

        return audio_buffer.read()
