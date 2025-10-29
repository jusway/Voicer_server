# handlers.py
import gradio as gr
from ASR.transcription_orchestrator import TranscriptionOrchestrator
from LLM.new_api_llm import NewApiLLM # 需要这个类型提示


def handle_asr_transcription(
        audio_file,
        context_text,
        *,
        transcriber: TranscriptionOrchestrator,
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
        save_outputs=False,
        progress=progress
    )
    return result["full_text"]


def handle_llm_list_models(llm_client: NewApiLLM):
    """
    处理“拉取模型列表”按钮点击事件。
    """
    print("Handler: 正在拉取模型列表...")
    models = llm_client.list_available_models()

    if not models:
        gr.Warning("未能从 API 获取到模型列表，请检查 Key 或中转站配置。")
        return gr.Dropdown(choices=[], value=None)

    # 自动选择一个默认模型（例如，包含 'gemini' 的第一个）
    default_model = next((m for m in models if 'gemini' in m.lower()), models[0])
    print(f"Handler: 成功拉取 {len(models)} 个模型，默认选择 {default_model}")

    return gr.Dropdown(choices=models, value=default_model)





