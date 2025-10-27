from abc import ABC, abstractmethod
class BaseASR(ABC):
    @abstractmethod
    def transcribe(self, audio_file_path: str, context: str) -> tuple[str, str]:
        # 返回 (language, text)
        pass