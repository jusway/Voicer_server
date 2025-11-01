from typing import List
from LLM.base_llm import ChatMessage
from LLM.message_builder import MessageBuilder


class VoiceProofreadOneShot:
    """
    语音转写校对的 one-shot 构造器：
    - 固定 system 提示词
    - 固定一个 user/assistant 的示例
    - 根据用户输入返回 OpenAI 风格 messages
    """

    SYSTEM_PROMPT: str = (
        "你是一个精准的中文语音转写校对助手。"
        "请对口语化转写文本进行校对，不改变原本语义："
        "1) 修正明显的错别字、断句与标点；"
        "2) 适度移除口头填充词（如“嗯”“啊”“那个”），不删改有效信息；"
        "3) 统一数字、时间等书写；"
        "4) 不凭空添加或扩写；"
        "输出仅为校对后的纯文本。"
    )

    # one-shot 示例（user -> assistant）
    ONE_SHOT_USER: str = (
        "今天早上八点半我去，嗯，那个医院看病然后医生说要我十二点，再回来然后我就啊，回家了"
    )
    ONE_SHOT_ASSISTANT: str = (
        "今天早上八点半我去医院看病，然后医生让我十二点再回来，之后我就回家了。"
    )

    def build_messages(self, user_prompt: str) -> List[ChatMessage]:
        """
        根据用户输入构造 OpenAI 风格的 messages。
        顺序为：system -> one-shot(user) -> one-shot(assistant) -> user(真实输入)
        """
        builder = MessageBuilder(system_prompt=self.SYSTEM_PROMPT)
        builder.add_user(self.ONE_SHOT_USER)
        builder.add_assistant(self.ONE_SHOT_ASSISTANT)
        builder.add_user(user_prompt)
        return builder.build()