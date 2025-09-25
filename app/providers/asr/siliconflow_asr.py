from .base import ASRProvider

class SiliconFlowASR(ASRProvider):
    name = "siliconflow"

    def transcribe(self, wav_path: str, lang: str = "zh") -> str:
        raise NotImplementedError

