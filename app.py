# app.py 负责前端的构建和启动
import gradio as gr
import functools
import os
import config  # 必须首先导入，以加载 .env
import handlers
from ASR.transcription_orchestrator import TranscriptionOrchestrator
from ASR.qwen_asr import QwenASR
from LLM.new_api_llm import NewApiLLM


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
            "New API": NewApiLLM(
                base_url=config.NEW_API_BASE_URL,
                api_key=config.NEW_API_KEY,
                user_id=config.NEW_API_USER_ID
            )
        }

        print("所有服务初始化完毕。")

    def build_ui(self):
        """
        构建Gradio UI。
        """
        with gr.Blocks(title="声稿师 Voicer") as app:
            with gr.Tabs() as level1_tabs:

                # --- 第一层标签: 语音识别 ---
                with gr.TabItem("语音识别"):
                    with gr.Tabs() as asr_level_tabs:
                        # 动态创建第二层标签
                        for name, transcriber_instance in self.transcribers.items():
                            with gr.TabItem(name):
                                self._build_asr_tab(transcriber_instance)

                with gr.TabItem("大模型"):
                    with gr.Tabs() as llm_level_tabs:
                        # 动态创建第二层标签
                        for name, llm_client_instance in self.llm_clients.items():
                            if isinstance(llm_client_instance, NewApiLLM):
                                with gr.TabItem("New API 接口"):
                                    # 调用新的辅助方法
                                    self._build_llm_tab(llm_client_instance)

        return app

    def _build_asr_tab(self, transcriber: TranscriptionOrchestrator):
        """辅助方法：创建一个ASR标签页的UI布局"""

        with gr.Row():
            with gr.Column(scale=1, min_width=250):
                file_input = gr.File(
                    label="上传音频/视频文件",
                    height=200
                )

            with gr.Column(scale=3, min_width=400):
                context_input = gr.Textbox(
                    label="上下文提示词 (可选)",
                    lines=7
                )

        examples_data = [
            ["一位法师在讲经说法。热词：羯磨。"],
        ]
        gr.Examples(
            examples=examples_data,
            inputs=[context_input],
            label="提示词案例 (点击自动填充)",
            examples_per_page=5  # 每页显示多少个案例
        )

        submit_btn = gr.Button("开始转录", variant="primary")

        with gr.Column():
            text_output = gr.Textbox(label="转录结果", lines=15)

        # 将 'fn' 绑定到 'handlers.py' 中的函数
        handler = functools.partial(
            handlers.handle_asr_transcription,
            transcriber=transcriber
        )

        submit_btn.click(
            fn=handler,
            inputs=[file_input, context_input],
            outputs=[text_output]
        )

    def _build_llm_tab(self, llm_client: NewApiLLM):
        """
        辅助方法：创建一个 LLM 聊天标签页的UI布局
        (第一阶段：只实现拉取模型列表)
        """

        with gr.Row():
            get_models_btn = gr.Button("刷新模型列表", variant="primary", scale=1)
            model_dropdown = gr.Dropdown(
                label="可用的模型",
                interactive=True,
                scale=3,
                info="列表将在此处显示"
            )

        # 绑定"拉取模型列表"
        list_models_handler = functools.partial(
            handlers.handle_llm_list_models,
            llm_client=llm_client
        )

        get_models_btn.click(
            fn=list_models_handler,
            inputs=[],  # 没有输入
            outputs=[model_dropdown]  # 输出到下拉框
        )


    def launch(self):
        """构建UI并启动Gradio服务"""
        # (这个方法在你的当前结构中似乎未被使用，但我们保留它)
        self.app = self.build_ui()
        self.app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            # share=True,
            debug=True,
        )


# (全局实例和 if __name__ == "__main__" 部分保持不变)
app_state = App()
demo = app_state.build_ui()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
    )