from abc import ABC, abstractmethod
from typing import Generator, Optional, List, Dict

# 定义历史记录中单个消息的类型别名，更清晰
ChatMessage = Dict[str, str] # 通常包含 'role' 和 'content'

class BaseLLM(ABC):
    """
    LLM 服务的抽象基类 (Abstract Base Class)。
    定义了一个通用的、流式优先的聊天接口，适用于各种 LLM 任务。
    """

    @abstractmethod
    def chat_stream(self,
                    user_input: str,
                    history: Optional[List[ChatMessage]] = None,
                    system_prompt: Optional[str] = None,
                    model: Optional[str] = None,
                    **kwargs) -> Generator[str, None, None]:
        """
        (流式) 与 LLM 进行交互或生成文本。
        *子类必须实现此方法以支持流式响应。*

        :param user_input: 当前用户的输入或指令 (str)。
        :param history: (可选) 之前的对话历史记录列表 (List[ChatMessage])。
                       每个 ChatMessage 是一个字典，通常包含 'role' 和 'content'。
        :param system_prompt: (可选) 给模型的系统级指令 (str)。
                              注意：如何结合 system_prompt 和 history 取决于具体 API 的要求。
                              子类实现可能需要将 system_prompt 格式化为 history 的一部分。
        :param model: (可选) 指定要使用的具体模型名称 (str)，如果 API 支持或需要。
        :param kwargs: (可选) 允许传入特定 API 可能需要的额外参数 (例如 temperature, max_tokens)。
                       子类实现应根据需要处理这些参数。
        :return: 一个生成器，逐块 (chunk) yield LLM 生成的文本 (str)。
        """
        # 这个 yield 语句仅用于满足 Python 解释器和类型检查器
        if False: # pragma: no cover
            yield


