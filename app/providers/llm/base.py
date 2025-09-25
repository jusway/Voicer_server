class LLMProvider:
    name: str = "base"

    def polish(self, text: str, prompt: str, **kwargs) -> str:
        raise NotImplementedError

