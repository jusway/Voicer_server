# handlers.py
import gradio as gr
from ASR.transcription_orchestrator import LongTextTranscriptionOrchestrator
from LLM.new_api_llm import NewApiLLM # 需要这个类型提示
from LLM.message_builder import MessageBuilder
from LLM.voice_proofread_one_shot import VoiceProofreadOneShot
from LLM.proofread_comparator import ProofreadComparator
from typing import Optional, Generator


def handle_asr_transcription(
        audio_file,
        context_text,
        *,
        transcriber: LongTextTranscriptionOrchestrator,
        progress=gr.Progress(track_tqdm=True)
):
    """
    处理 ASR 转录的后端逻辑。
    'transcriber' 参数将由 functools.partial 传入。
    """
    if audio_file is None:
        return "错误：请先上传文件。"

    print(f"收到任务: 使用 {transcriber.asr_client.__class__.__name__} 进行转录...")

    result = transcriber.transcribe(
        input_file=audio_file.name,
        context=context_text,
        progress=progress
    )
    return result["full_text"]


def handle_llm_list_models(llm_client: NewApiLLM):
    """
    处理"拉取模型列表"按钮点击事件。
    """
    print("Handler: 正在拉取模型列表...")
    models = llm_client.list_available_models()

    if not models:
        gr.Warning("未能从 API 获取到模型列表，请检查 Key 或中转站配置。")
        return gr.Dropdown(choices=[], value=None)

    # 默认选择第一个模型
    default_model = models[0]
    print(f"Handler: 成功拉取 {len(models)} 个模型，默认选择 {default_model}")

    return gr.Dropdown(choices=models, value=default_model)



def handle_llm_chat_stream(
        system_prompt: Optional[str],
        user_input: str,
        selected_model: Optional[str],
        *,
        llm_client: NewApiLLM,
) -> Generator[str, None, None]:
    """
    流式聊天处理器：
    - 将系统提示词与当前输入组装为 messages
    - 调用 llm_client.chat_stream 并逐步返回累积文本
    - 输出为单个文本框的流式内容
    """
    partial_text = ""

    # 基本校验
    if not selected_model:
        gr.Warning("请先选择模型")
        yield "错误：未选择模型。"
        return
    if not isinstance(user_input, str) or not user_input.strip():
        gr.Warning("请输入当前用户输入")
        yield "错误：当前用户输入为空。"
        return

    builder = MessageBuilder(system_prompt=system_prompt, base_messages=None)
    builder.add_user(user_input)
    messages = builder.build()

    try:
        for chunk in llm_client.chat_stream(messages=messages, model=selected_model):
            if chunk:
                partial_text += chunk
                yield partial_text
        # 结束时，输出最终文本
        yield partial_text
    except Exception as e:
        gr.Warning(f"生成失败：{e}")
        yield f"错误：{e}"



def handle_proofread_stream(
        user_transcript: str,
        selected_model: Optional[str],
        *,
        llm_client: NewApiLLM,
) -> Generator[str, None, None]:
    """
    语音稿校对的流式处理器：
    - 使用固定的系统提示词与 one-shot 示例构造 messages
    - 调用 llm_client.chat_stream 并逐步返回累积文本
    - 输出为单个文本框的流式内容
    """
    partial_text = ""

    # 基本校验
    if not selected_model:
        gr.Warning("请先选择模型")
        yield "错误：未选择模型。"
        return
    if not isinstance(user_transcript, str) or not user_transcript.strip():
        gr.Warning("请输入待校对的语音稿")
        yield "错误：待校对语音稿为空。"
        return

    builder = VoiceProofreadOneShot()
    messages = builder.build_messages(user_transcript)

    try:
        for chunk in llm_client.chat_stream(messages=messages, model=selected_model):
            if chunk:
                partial_text += chunk
                yield partial_text
        yield partial_text
    except Exception as e:
        gr.Warning(f"校对失败：{e}")
        yield f"错误：{e}"


def handle_proofread_compare(original_text: str, corrected_text: str):
    """
    比较原始语音稿与校对稿，返回单视图 HTML：
    - 删除片段浅红背景、高亮插入位置
    - 新增片段浅绿背景
    - 替换为红+绿组合
    """
    if not isinstance(original_text, str) or not original_text.strip():
        gr.Warning("原始语音稿为空")
        return "错误：原始语音稿为空。"
    if not isinstance(corrected_text, str) or not corrected_text.strip():
        gr.Warning("校对结果为空")
        return "错误：校对结果为空。"

    comparator = ProofreadComparator(granularity="char")
    md_single = comparator.render_html_singleview_highlight(original_text, corrected_text)
    return md_single





