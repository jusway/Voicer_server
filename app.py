# app.py 负责前端的构建和启动
import gradio as gr
import functools
import config  # 必须首先导入，以加载 .env
import handlers
from ASR.transcription_orchestrator import TranscriptionOrchestrator
from ASR.qwen_asr import QwenASR


class App:

    def __init__(self):
        """
        初始化函数：加载所有模型和客户端，一次性完成。
        所有状态都存储在 'self' 中。
        """
        print("正在初始化所有服务...")

        # --- 初始化所有独立的 "引擎" ---
        self.qwen_engine = QwenASR("qwen3-asr-flash")
        # self.siliconflow_engine = SiliconFlowASR() # 示例

        # --- ASR: 为每个引擎创建专用的长音频 "编排器" ---
        self.transcribers = {
            "千问语音识别": TranscriptionOrchestrator(asr_client=self.qwen_engine, num_threads=4),
            # "硅基流动": TranscriptionOrchestrator(asr_client=self.siliconflow_engine, num_threads=4),
        }

        # --- LLM: 引擎比较简单，直接聚合 ---
        self.llm_clients = {
            # "DeepSeek": DeepSeekLLM(), # 示例
        }

        print("所有服务初始化完毕。")

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
                                # UI 布局本身不变
                                self._build_asr_tab(transcriber_instance)

        # (您未来可以在这里添加 "文本处理" 的 TabItem)

        return app

    def _build_asr_tab(self, transcriber: TranscriptionOrchestrator):
        """辅助方法：创建一个ASR标签页的UI布局"""
        with gr.Row():
            file_input = gr.File(label="上传音频/视频文件")
            context_input = gr.Textbox(label="上下文提示词 (可选)")

        submit_btn = gr.Button("开始转录", variant="primary")

        with gr.Column():
            # lang_output = gr.Textbox(label="检测到的语言") # <-- 已移除
            text_output = gr.Textbox(label="转录结果", lines=15)

        # 将 'fn' 绑定到 'handlers.py' 中的函数
        handler = functools.partial(handlers.handle_asr_transcription, transcriber=transcriber)

        submit_btn.click(
            fn=handler,
            inputs=[file_input, context_input],
            outputs=[text_output]  # <-- 已更新：只输出到 text_output
        )

    def launch(self):
        """构建UI并启动Gradio服务"""
        self.app = self.build_ui()
        self.app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            # share=True,
        )


if __name__ == "__main__":
    app = App()
    app.launch()
