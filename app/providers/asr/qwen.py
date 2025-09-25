from .base import ASRProvider

class QwenASR(ASRProvider):
    name = "qwen"

    def transcribe(self, wav_path: str, lang: str = "zh") -> str:
        raise NotImplementedError

