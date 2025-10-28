# LLM/new_api_llm.py
import requests
import json
from typing import Generator, Optional, List, Dict, Any
from urllib.parse import quote, urljoin
from LLM.base_llm import BaseLLM, ChatMessage


class NewApiLLM(BaseLLM):
    def __init__(self,
                 base_url: str,
                 api_key: str,
                 user_id: str,
                 default_model: Optional[str] = None):
        if not base_url: raise ValueError("base_url 不能为空")
        if not api_key: raise ValueError("api_key 不能为空")
        if not user_id: raise ValueError("user_id 不能为空")

        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.user_id = user_id
        self.default_model = default_model
        self._available_models: Optional[List[str]] = None

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
        })
        print(f"NewApi LLM Client Initialized. Base URL: {self.base_url}, User ID: {self.user_id}")

    def list_available_models(self, force_refresh: bool = False) -> List[str]:
        """
        (新版本) 尝试使用 OpenAI 兼容的 /v1/models 接口获取模型列表。
        """
        if self._available_models is not None and not force_refresh:
            print("从缓存返回可用模型列表。")
            return self._available_models

        # 关键改动 1: URL 变了
        models_url = f"{self.base_url}/v1/models"
        model_names = []

        print(f"正在从 (OpenAI 兼容接口) {models_url} 拉取可用模型列表...")

        headers = self._session.headers.copy()  # 包含 Authorization
        # 注意：标准的 /v1/ 接口不需要 New-Api-User 头部

        try:
            response = self._session.get(models_url, headers=headers, timeout=10)

            # 如果是 401 (access token 无效), 这里会直接抛出异常
            response.raise_for_status()

            data = response.json()

        except requests.exceptions.HTTPError as http_err:
            # 尝试解析 401 错误体，它可能是 OpenAI 格式的
            try:
                error_data = http_err.response.json()
                # 尝试提取 OpenAI 风格的错误信息
                message = error_data.get('error', {}).get('message', str(http_err))
            except json.JSONDecodeError:
                # 如果返回的不是 JSON (比如纯文本 401)
                message = str(http_err)

            raise RuntimeError(f"API 请求失败 (HTTP {http_err.response.status_code}): {message}")
        except Exception as e:
            raise RuntimeError(f"请求时发生未知错误: {e}")

        # 关键改动 2: 解析 OpenAI 格式的响应
        # 标准响应格式: { "object": "list", "data": [ {"id": "model-1", ...}, {"id": "model-2", ...} ] }
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            model_data = data['data']
            for model_info in model_data:
                if isinstance(model_info, dict) and 'id' in model_info:
                    model_names.append(model_info['id'])
        else:
            # 如果 Key 对了，但不符合 OpenAI 格式，也要报错
            raise RuntimeError(f"API 返回了非预期的 OpenAI 格式。Response: {data}")

        if not model_names: print("警告：未从 API 获取到任何模型名称。")

        unique_model_names = sorted(list(set(model_names)))
        self._available_models = unique_model_names
        print(f"成功获取并缓存了 {len(unique_model_names)} 个唯一模型。")
        return unique_model_names

    def _build_payload(self, user_input: str, history: Optional[List[ChatMessage]] = None, system_prompt: Optional[str] = None, model: str = "default", stream: bool = True, **kwargs) -> Dict[str, Any]:
        target_model = model if model != "default" else self.default_model
        if not target_model: raise ValueError("必须指定模型名称")
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        if history: messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        payload = {"model": target_model, "messages": messages, "stream": stream, **kwargs}
        return {k: v for k, v in payload.items() if v is not None}

    def chat_stream(self, user_input: str, history: Optional[List[ChatMessage]] = None, system_prompt: Optional[str] = None, model: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        request_model = model or self.default_model
        if not request_model: raise ValueError("必须指定模型名称")

        payload = self._build_payload(user_input, history, system_prompt, request_model, True, **kwargs)
        # 确认聊天端点 (假设是 /v1/chat/completions)
        request_url = f"{self.base_url}/v1/chat/completions" # <-- 如果您的 base_url 不含 /v1 了，这里要改回

        headers = self._session.headers.copy() # 包含 Authorization
        headers["Content-Type"] = "application/json" # 确保存在
        headers["Accept"] = "text/event-stream"
        headers['New-Api-User'] = self.user_id # 使用原始 User ID

        response = self._session.post(request_url, json=payload, headers=headers, stream=True, timeout=180)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    json_str = decoded_line[len('data: '):].strip()
                    if json_str == '[DONE]': break
                    if not json_str: continue
                    try:
                        data = json.loads(json_str)
                        choices = data.get("choices")
                        if choices and len(choices) > 0:
                            delta = choices[0].get("delta")
                            if delta:
                                content = delta.get("content")
                                if content: yield content
                    except json.JSONDecodeError:
                         print(f"警告：无法解析 SSE 数据块: {json_str}")
                         continue


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv

    load_dotenv()
    base_url = os.getenv("NEW_API_BASE_URL")
    api_key = os.getenv("NEW_API_KEY")
    user_id = os.getenv("NEW_API_USER_ID")

    if not base_url or not api_key or not user_id:
        print("请在 .env 文件中设置 NEW_API_BASE_URL, NEW_API_KEY 和 NEW_API_USER_ID")
    else:
        relay_client = NewApiLLM(base_url=base_url, api_key=api_key, user_id=user_id)
        available_models = relay_client.list_available_models()
        print("\n--- 可用模型列表 ---")
        if available_models:
            for m in available_models: print(f"- {m}")
        else:
            print("未能获取到模型列表。")
        print("--------------------\n")

        if available_models:
            test_model = next((m for m in available_models if 'qwen' in m.lower()), None)
            if test_model:
                print(f"--- 使用模型 '{test_model}' 进行测试 ---")
                user_query = "你好，用中文介绍一下你自己。"
                print(f"用户: {user_query}")
                print("助手 (流式): ", end="")
                for chunk in relay_client.chat_stream(user_input=user_query, model=test_model):
                    print(chunk, end="", flush=True)
                print("\n--------------------")
            else:
                print("在中转站可用模型中未找到合适的测试模型（如 qwen），跳过聊天测试。")
        else:
            print("无法获取模型列表，跳过聊天测试。")

