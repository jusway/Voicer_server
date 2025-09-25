from .base import LLMProvider

class SiliconFlowLLM(LLMProvider):
    name = "siliconflow"

    def polish(self, text: str, prompt: str, **kwargs) -> str:
        raise NotImplementedError

