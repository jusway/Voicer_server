import os
import requests
import concurrent.futures
import shutil
from tqdm import tqdm
from collections import Counter
from urllib.parse import urlparse
import gradio as gr

# 用于类型提示
from ASR.base_asr import BaseASR

from ASR.audio_tools import load_audio, process_vad, save_audio_file, WAV_SAMPLE_RATE
from silero_vad import load_silero_vad


class TranscriptionOrchestrator:
    """
    一个面向对象的长音频转录“编排器”。
    它 *使用* 一个抽象的 BaseASR 实例来执行转录。
    封装了音频加载、VAD分段、并行API请求、结果合成和文件保存的完整流程。
    """

    def __init__(self,
                 asr_client: BaseASR,  # <-- 明确依赖抽象
                 num_threads: int = 4,
                 vad_segment_threshold_s: int = None,  # <-- 默认值为 None
                 tmp_dir: str = os.path.join(os.path.expanduser("~"), "qwen3-asr-cache")):
        """
        初始化长音频转录编排器。

        :param asr_client: 一个 *已经配置好* 的、实现了 BaseASR 接口的客户端实例。
        :param num_threads: 并行处理的线程数。
        :param vad_segment_threshold_s: VAD 分割音频的目标时长（秒）。
                                          如果为 None, 将自动设为引擎最大时长的 90%。
        :param tmp_dir: 存放临时音频片段的目录。
        """
        if not isinstance(asr_client, BaseASR):
            raise TypeError(f"asr_client must be an instance of BaseASR, but got {type(asr_client)}")

        self.asr_client = asr_client  # 依赖注入
        self.num_threads = num_threads
        self.tmp_dir = tmp_dir
        self.vad_model = None  # VAD 模型将按需加载

        # 从 ASR 客户端获取其最大时长限制
        self.client_max_duration_s = asr_client.max_duration_seconds

        # 智能设置 VAD 分割阈值
        if vad_segment_threshold_s is None:
            # 如果用户未指定，默认设为客户端最大时长的 90% (留出安全边际)
            self.vad_segment_threshold = int(self.client_max_duration_s * 0.9)
            print(f"VAD threshold not set. Defaulting to 90% of client limit: {self.vad_segment_threshold}s")
        elif vad_segment_threshold_s > self.client_max_duration_s:
            # 如果用户设置过高，警告并强制设为最大值
            print(
                f"Warning: Provided VAD threshold ({vad_segment_threshold_s}s) exceeds client limit ({self.client_max_duration_s}s).")
            print(f"Clamping VAD threshold to {self.client_max_duration_s}s.")
            self.vad_segment_threshold = self.client_max_duration_s
        else:
            # 使用用户提供的安全值
            self.vad_segment_threshold = vad_segment_threshold_s

    def _load_vad_model(self):
        """延迟加载 Silero VAD 模型以节省资源。"""
        if self.vad_model is None:
            print("Initializing Silero VAD model for segmenting...")
            self.vad_model = load_silero_vad(onnx=True)

    def _validate_input(self, file_path: str):
        """检查输入文件或URL是否存在且可访问。"""
        if file_path.startswith(("http://", "https://")):
            try:
                response = requests.head(file_path, allow_redirects=True, timeout=5)
                if response.status_code >= 400:
                    raise FileNotFoundError(f"URL returned status code {response.status_code}")
            except Exception as e:
                raise FileNotFoundError(f"HTTP link {file_path} does not exist or is inaccessible: {e}")
        elif not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file \"{file_path}\" does not exist!")

    def _prepare_audio_segments(self, file_path: str) -> list:
        """加载、标准化并智能分段音频。"""
        wav = load_audio(file_path)
        duration_s = len(wav) / WAV_SAMPLE_RATE
        print(f"Loaded wav duration: {duration_s:.2f}s")

        # 使用动态获取的客户端最大时长
        if duration_s >= self.client_max_duration_s:
            print(f"Duration exceeds client limit ({self.client_max_duration_s}s), initiating VAD...")
            self._load_vad_model()

            # 将 VAD 的“目标时长”和“最大时长”都传递给 process_vad
            #    （假设 process_vad 接受 max_segment_threshold_s 参数，
            #     正如您最早提供的代码那样）
            wav_list = process_vad(
                wav,
                self.vad_model,
                segment_threshold_s=self.vad_segment_threshold,
                max_segment_threshold_s=self.client_max_duration_s
            )

            print(f"Segmenting done, total segments: {len(wav_list)}")
        else:
            wav_list = [(0, len(wav), wav)]
        return wav_list

    def _create_tmp_chunks(self, wav_list: list, input_file: str) -> tuple[list, str]:
        """将音频分段保存为临时的 .wav 文件。"""
        wav_name = os.path.basename(input_file)
        wav_dir_name = os.path.splitext(wav_name)[0]
        save_dir = os.path.join(self.tmp_dir, wav_dir_name)

        wav_path_list = []
        for idx, (_, _, wav_data) in enumerate(wav_list):
            wav_path = os.path.join(save_dir, f"{wav_name}_{idx}.wav")
            save_audio_file(wav_data, wav_path)
            wav_path_list.append(wav_path)
        return wav_path_list, save_dir

    def _run_parallel_transcription(self, wav_path_list: list, context: str) -> tuple[list, list]:
        """使用线程池并行执行 ASR 任务。"""
        results = []
        languages = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_dict = {
                executor.submit(self.asr_client.transcribe, wav_path, context): idx
                for idx, wav_path in enumerate(wav_path_list)
            }
            with tqdm(total=len(future_dict), desc="Calling ASR API") as pbar:
                for future in concurrent.futures.as_completed(future_dict):
                    idx = future_dict[future]
                    try:
                        language, recog_text = future.result()
                        results.append((idx, recog_text))
                        languages.append(language)
                    except Exception as e:
                        print(f"Error processing segment {idx}: {e}")
                        results.append((idx, f"[ERROR: {e}]"))
                        languages.append("Unknown")
                    pbar.update(1)
        return results, languages

    def _compose_results(self, raw_results: list, languages: list) -> dict:
        """合成最终的文本和语言。"""
        raw_results.sort(key=lambda x: x[0])
        full_text = " ".join(text for _, text in raw_results)
        language = Counter(languages).most_common(1)[0][0] if languages else "Unknown"

        return {
            "language": language,
            "full_text": full_text,
        }

    def _save_results(self, results: dict, input_file: str, output_dir: str) -> dict:
        """将结果保存到 .txt 文件。"""
        if os.path.exists(input_file):
            base_name = os.path.splitext(os.path.basename(input_file))[0]
        else:
            base_name = os.path.splitext(urlparse(input_file).path)[0].split('/')[-1]

        output_dir = output_dir or os.path.dirname(input_file) or "."
        os.makedirs(output_dir, exist_ok=True)

        txt_file_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(results["language"] + '\n')
            f.write(results["full_text"] + '\n')
        print(f"Full transcription saved to \"{txt_file_path}\"")

        return {"txt_file_path": txt_file_path}

    def _cleanup(self, tmp_save_dir: str):
        """删除临时文件目录。"""
        if os.path.exists(tmp_save_dir):
            shutil.rmtree(tmp_save_dir)
            print(f"Cleaned up temporary directory: {tmp_save_dir}")

    def transcribe(self,
                   input_file: str,
                   context: str = "",
                   save_outputs: bool = True,
                   output_dir: str = None,
                   cleanup_tmp: bool = True,
                   progress: gr.Progress = None
                   ) -> dict:
        """
        执行完整的长音频转录流程。
        """
        self._validate_input(input_file)

        tmp_save_dir = None
        try:
            progress(0.1, desc="音频准备中...VAD过程...")
            wav_list = self._prepare_audio_segments(input_file)
            progress(0.2, desc="分段处理中...")
            wav_path_list, tmp_save_dir = self._create_tmp_chunks(wav_list, input_file)
            progress(0.3, desc="收到任务，开始调用API...平均5分钟音频需要30秒转录...")
            raw_results, languages = self._run_parallel_transcription(wav_path_list, context)
            progress(0.99, desc="处理完成，开始合成结果...")
            final_results = self._compose_results(raw_results, languages)

            if save_outputs:
                saved_paths = self._save_results(final_results, input_file, output_dir)
                final_results.update(saved_paths)

            return final_results

        finally:
            if cleanup_tmp and tmp_save_dir:
                self._cleanup(tmp_save_dir)


if __name__ == '__main__':
    # --- 使用示例 ---
    from ASR.qwen_asr import QwenASR
    import config
    qwen_client = QwenASR(model="qwen3-asr-flash")
    orchestrator = TranscriptionOrchestrator(
        asr_client=qwen_client,
        num_threads=4,
    )
    orchestrator.transcribe(input_file=r"D:\DATA\Voicer测试数据\如何学戒_五分钟.wav", save_outputs=True)
