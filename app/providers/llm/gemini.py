from .base import LLMProvider

class GeminiLLM(LLMProvider):
    name = "gemini"

    def polish(self, text: str, prompt: str, **kwargs) -> str:
        raise NotImplementedError

