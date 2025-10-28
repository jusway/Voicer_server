# handlers.py
import gradio as gr
from ASR.transcription_orchestrator import TranscriptionOrchestrator


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


