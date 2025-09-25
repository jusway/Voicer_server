from .base import LLMProvider

class DeepSeekLLM(LLMProvider):
    name = "deepseek"

    def polish(self, text: str, prompt: str, **kwargs) -> str:
        raise NotImplementedError

