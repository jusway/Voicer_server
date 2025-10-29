from typing import List, Optional
from LLM.base_llm import ChatMessage


class MessageBuilder:
    """
    轻量消息构造器，用于按 OpenAI 风格拼接 messages：
    - 可选 system 提示
    - 追加已有历史（user/assistant）
    - 追加当前输入
    - 校验 role 与 content
    """

    def __init__(self, system_prompt: Optional[str] = None, base_messages: Optional[List[ChatMessage]] = None):
        self._messages: List[ChatMessage] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})
        if base_messages:
            self.extend(base_messages)

    def extend(self, messages: List[ChatMessage]) -> "MessageBuilder":
        for m in messages:
            self._validate_message(m)
            self._messages.append(m)
        return self

    def add_user(self, content: str) -> "MessageBuilder":
        msg = {"role": "user", "content": content}
        self._validate_message(msg)
        self._messages.append(msg)
        return self

    def add_assistant(self, content: str) -> "MessageBuilder":
        msg = {"role": "assistant", "content": content}
        self._validate_message(msg)
        self._messages.append(msg)
        return self

    def build(self) -> List[ChatMessage]: 
        return list(self._messages)

    @staticmethod
    def _validate_message(m: ChatMessage) -> None:
        valid_roles = {"system", "user", "assistant"}
        role = m.get("role")
        content = m.get("content")
        if role not in valid_roles:
            raise ValueError(f"非法 role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content 必须是非空字符串")


