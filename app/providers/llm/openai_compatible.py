from .base import LLMProvider

class OpenAICompatibleLLM(LLMProvider):
    name = "openai-compatible"

    def polish(self, text: str, prompt: str, **kwargs) -> str:
        raise NotImplementedError

