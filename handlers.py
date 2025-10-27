# handlers.py
import gradio as gr
from ASR.transcription_orchestrator import TranscriptionOrchestrator


def handle_asr_transcription(
        # 1. 首先，列出 Gradio 'inputs' 对应的参数 (按顺序)
        audio_file,
        context_text,

        # 2. 添加一个单独的星号 '*'
        *,

        # 3. 在 '*' 后面列出所有“仅限关键字”的参数
        transcriber: TranscriptionOrchestrator,
        progress=gr.Progress(track_tqdm=True)
):
    """
    处理 ASR 转录的后端逻辑。
    'transcriber' 参数将由 functools.partial 传入。
    （已更新：只返回一个字符串结果）
    """
    if audio_file is None:
        return "错误：请先上传文件。"  # <-- 只返回一个值

    print(f"收到任务: 使用 {transcriber.asr_client.__class__.__name__} 进行转录...")

    try:
        result = transcriber.transcribe(
            input_file=audio_file.name,
            context=context_text,
            save_outputs=False  # Gradio 应用总是在 UI 上显示结果
        )
        return result["full_text"]  # <-- 只返回一个值

    except Exception as e:
        print(f"转录过程中出错: {e}")
        # 将错误信息返回给 UI 界面
        return f"转录失败: {str(e)}"  # <-- 只返回一个值
