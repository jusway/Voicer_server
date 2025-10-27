import gradio as gr
import functools
import config  # 必须首先导入，以加载 .env
from ASR.transcription_orchestrator import LongAudioTranscriber
from ASR.qwen_asr import QwenASR



class MultiServiceApp:

    def __init__(self):
        """
        初始化函数：加载所有模型和客户端，一次性完成。
        所有状态都存储在 'self' 中。
        """
        print("正在初始化所有服务...")

        # --- 初始化所有独立的 "引擎" ---
        self.qwen_engine = QwenASR("qwen3-asr-flash")

        # --- ASR: 为每个引擎创建专用的长音频 "编排器" ---
        self.transcribers = {
            "千问语音识别": LongAudioTranscriber(asr_client=self.qwen_engine, num_threads=4),
        }

        # --- LLM: 引擎比较简单，直接聚合 ---
        self.llm_clients = {
            "中转站api": self.relay_llm_engine,
            "deepseek的api": self.deepseek_llm_engine
            # 未来在这里添加更多
        }

        print("所有服务初始化完毕。")

    # --- UI 构建方法 ---

    def build_ui(self):
        """
        构建Gradio UI。
        """
        with gr.Blocks(title="智能助手") as app:
            gr.Markdown("# 智能助手")

            with gr.Tabs() as level1_tabs:

                # --- 第一层标签: 语音识别 ---
                with gr.TabItem("语音识别"):
                    with gr.Tabs() as asr_level2_tabs:
                        # 动态创建第二层标签
                        for name, transcriber_instance in self.transcribers.items():
                            with gr.TabItem(name):
                                self._build_asr_tab(transcriber_instance)

                # --- 第一层标签: 文本处理 ---
                with gr.TabItem("文本处理"):
                    with gr.Tabs() as llm_level2_tabs:
                        # 动态创建第二层标签
                        for name, llm_client_instance in self.llm_clients.items():
                            with gr.TabItem(name):
                                self._build_llm_tab(llm_client_instance)
        return app

    def _build_asr_tab(self, transcriber: LongAudioTranscriber):
        """辅助方法：创建一个ASR标签页的UI布局"""
        with gr.Row():
            file_input = gr.File(label="上传音频/视频文件")
            context_input = gr.Textbox(label="上下文提示词 (可选)")

        submit_btn = gr.Button("开始转录", variant="primary")

        with gr.Column():
            lang_output = gr.Textbox(label="检测到的语言")
            text_output = gr.Textbox(label="转录结果", lines=15)

        # 关键：将按钮的点击事件，绑定到 *特定* 的处理函数
        # 我们使用 functools.partial 来 "固定" 这个标签页对应的 transcriber 实例
        handler = functools.partial(self.handle_asr_transcription, transcriber=transcriber)

        submit_btn.click(
            fn=handler,
            inputs=[file_input, context_input],
            outputs=[lang_output, text_output]
        )

    def _build_llm_tab(self, llm_client):
        """辅助方法：创建一个LLM标签页的UI布局"""
        with gr.Row():
            text_input = gr.Textbox(label="原文", lines=15, scale=2)
            with gr.Column(scale=1):
                submit_btn = gr.Button("开始润色/校对", variant="primary")
                text_output = gr.Textbox(label="润色/校对后", lines=15)

        diff_output = gr.HTML(label="文稿对比 (Diff View)")

        # 同样，使用 partial 来 "固定" 这个标签页对应的 llm_client 实例
        handler = functools.partial(self.handle_llm_processing, client=llm_client)

        submit_btn.click(
            fn=handler,
            inputs=[text_input],
            outputs=[text_output, diff_output]
        )

    # --- 事件处理方法 ---

    def handle_asr_transcription(self, transcriber: LongAudioTranscriber, audio_file, context_text,
                                 progress=gr.Progress(track_tqdm=True)):
        """
        处理 ASR 转录的后端逻辑。
        'transcriber' 参数是由 functools.partial 传入的。
        """
        if audio_file is None:
            return "错误", "请先上传文件。"

        print(f"收到任务: 使用 {transcriber.asr_client.__class__.__name__} 进行转录...")
        try:
            result = transcriber.transcribe(
                input_file=audio_file.name,
                context=context_text,
                save_outputs=False
            )
            return result["language"], result["full_text"]
        except Exception as e:
            return "错误", f"转录失败: {e}"

    def handle_llm_processing(self, client, text_input, progress=gr.Progress(track_tqdm=True)):
        """
        处理 LLM 润色的后端逻辑。
        'client' 参数是由 functools.partial 传入的。
        """
        if not text_input.strip():
            return "错误：原文不能为空。", ""

        print(f"收到任务: 使用 {client.__class__.__name__} 进行润色...")
        try:
            polished_text = client.polish_text(text_input)
            diff_html = self.diff_tool(text_input, polished_text)
            return polished_text, diff_html
        except Exception as e:
            return f"处理失败: {e}", ""

    # --- 启动方法 ---

    def launch(self):
        """构建UI并启动Gradio服务"""
        self.app = self.build_ui()
        self.app.launch(
            server_name="0.0.0.0",
            server_port=7860
        )


# --- 脚本入口 ---
if __name__ == "__main__":
    service_app = MultiServiceApp()
    service_app.launch()