class ASRProvider:
    name: str = "base"

    def transcribe(self, wav_path: str, lang: str = "zh") -> str:
        raise NotImplementedError

