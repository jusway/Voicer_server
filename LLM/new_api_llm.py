from typing import Generator, Optional, List, Dict
from openai import OpenAI
from LLM.base_llm import BaseLLM, ChatMessage


class NewApiLLM(BaseLLM):
    def __init__(self,
                 base_url: str,
                 api_key: str,
                 default_model: Optional[str] = None):
        if not base_url: raise ValueError("base_url 不能为空")
        if not api_key: raise ValueError("api_key 不能为空")

        base = base_url.rstrip('/')
        self.base_url = base if base.endswith('/v1') else (base + '/v1')
        self.api_key = api_key
        self.default_model = default_model
        self._available_models: Optional[List[str]] = None

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        print(f"NewApi LLM Client Initialized. Base URL: {self.base_url}")

    def list_available_models(self, force_refresh: bool = False) -> List[str]:
        if self._available_models is not None and not force_refresh:
            print("从缓存返回可用模型列表。")
            return self._available_models

        print("正在从 OpenAI 兼容接口拉取可用模型列表...")

        try:
            result = self._client.models.list()
        except Exception as e:
            raise RuntimeError(f"请求模型列表失败: {e}")

        model_names: List[str] = []
        try:
            data = getattr(result, "data", None)
            if isinstance(data, list):
                for m in data:
                    mid = getattr(m, "id", None)
                    if isinstance(mid, str):
                        model_names.append(mid)
        except Exception:
            pass

        if not model_names:
            print("警告：未从 API 获取到任何模型名称。")

        unique_model_names = sorted(list(set(model_names)))
        self._available_models = unique_model_names
        print(f"成功获取并缓存了 {len(unique_model_names)} 个唯一模型。")
        return unique_model_names

    def _select_model(self, model: Optional[str]) -> str:
        target_model = model or self.default_model
        if not target_model:
            raise ValueError("必须指定模型名称")
        return target_model

    def _validate_messages(self, messages: List[ChatMessage]) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages 必须是非空列表")
        valid_roles = {"system", "user", "assistant"}
        for i, m in enumerate(messages):
            if not isinstance(m, dict):
                raise ValueError(f"第 {i} 条消息不是字典: {m}")
            role = m.get("role")
            content = m.get("content")
            if role not in valid_roles:
                raise ValueError(f"第 {i} 条消息 role 非法: {role}")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"第 {i} 条消息 content 非法或为空")

    def chat_stream(self, messages: List[ChatMessage], model: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        target_model = self._select_model(model)
        self._validate_messages(messages)

        try:
            stream = self._client.chat.completions.create(
                model=target_model,
                messages=messages,
                stream=True,
                **kwargs,
            )
        except Exception as e:
            try:
                resp = self._client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    stream=False,
                    **kwargs,
                )
                content = None
                choices = getattr(resp, "choices", None)
                if choices:
                    first = choices[0]
                    message = getattr(first, "message", None)
                    if message is not None:
                        content = getattr(message, "content", None)
                if isinstance(content, str) and content:
                    yield content
                    return
                raise RuntimeError("非流式回退成功但未返回文本内容")
            except Exception as e2:
                raise RuntimeError(f"创建流式会话失败: {e}; 非流式回退也失败: {e2}")

        for chunk in stream:
            try:
                choices = getattr(chunk, "choices", None)
                if choices:
                    first = choices[0]
                    delta = getattr(first, "delta", None)
                    if delta is not None:
                        content = getattr(delta, "content", None)
                        if isinstance(content, str) and content:
                            yield content
            except Exception:
                continue


if __name__ == '__main__':
    pass

