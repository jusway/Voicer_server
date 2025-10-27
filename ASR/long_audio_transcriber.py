import os
import requests
import dashscope
import concurrent.futures
import shutil
from tqdm import tqdm
from collections import Counter
from urllib.parse import urlparse
from ASR.base_asr import BaseASR
from ASR.qwen_asr import QwenASR
from ASR.audio_tools import load_audio, process_vad, save_audio_file, WAV_SAMPLE_RATE
from silero_vad import load_silero_vad


class LongAudioTranscriber(BaseASR):
    """
    一个面向对象的长音频转录器。
    封装了音频加载、VAD分段、并行API请求、结果合成和文件保存的完整流程。
    （已精简：移除了 SRT 相关功能）
    """

    def __init__(self,
                 asr_client,
                 api_key: str = None,
                 num_threads: int = 4,
                 vad_segment_threshold_s: int = 120,
                 tmp_dir: str = os.path.join(os.path.expanduser("~"), "qwen3-asr-cache")):
        """
        初始化长音频转录器。

        :param asr_client: 一个实现了 .asr(wav_url, context) 方法的 ASR 客户端实例。
        :param api_key: (可选) DashScope API 密钥。如果提供，将设置它。
        :param num_threads: 并行处理的线程数。
        :param vad_segment_threshold_s: VAD 分割音频的目标时长（秒）。
        :param tmp_dir: 存放临时音频片段的目录。
        """
        self.asr_client = asr_client
        self.num_threads = num_threads
        self.vad_segment_threshold = vad_segment_threshold_s
        self.tmp_dir = tmp_dir
        self.vad_model = None  # VAD 模型将按需加载

        if api_key:
            dashscope.api_key = api_key
        elif "DASHSCOPE_API_KEY" not in os.environ:
            raise ValueError(
                "DashScope API key not found. Please provide it via the 'api_key' parameter or set the 'DASHSCOPE_API_KEY' environment variable.")

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
        print(f"Loaded wav duration: {len(wav) / WAV_SAMPLE_RATE:.2f}s")

        if len(wav) / WAV_SAMPLE_RATE >= 180:  # 只有超过3分钟的长音频才需要VAD
            self._load_vad_model()
            wav_list = process_vad(wav, self.vad_model, segment_threshold_s=self.vad_segment_threshold)
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
                        results.append((idx, f"[ERROR: {e}]"))  # 记录错误而不是让整个任务失败
                        languages.append("Unknown")
                    pbar.update(1)
        return results, languages

    def _compose_results(self, raw_results: list, languages: list, wav_list: list) -> dict:
        """合成最终的文本和语言。（已精简：移除SRT生成）"""
        raw_results.sort(key=lambda x: x[0])
        full_text = " ".join(text for _, text in raw_results)
        language = Counter(languages).most_common(1)[0][0] if languages else "Unknown"

        # --- SRT 生成代码已移除 ---

        return {
            "language": language,
            "full_text": full_text,
            # "srt_content" 键已移除
        }

    def _save_results(self, results: dict, input_file: str, output_dir: str) -> dict:
        """将结果保存到 .txt 文件。（已精简：移除SRT保存）"""
        if os.path.exists(input_file):
            base_name = os.path.splitext(os.path.basename(input_file))[0]
        else:
            base_name = os.path.splitext(urlparse(input_file).path)[0].split('/')[-1]

        output_dir = output_dir or os.path.dirname(input_file) or "."
        os.makedirs(output_dir, exist_ok=True)

        txt_file_path = os.path.join(output_dir, f"{base_name}.txt")
        # srt_file_path 已移除

        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(results["language"] + '\n')
            f.write(results["full_text"] + '\n')
        print(f"Full transcription saved to \"{txt_file_path}\"")

        # --- SRT 保存代码已移除 ---

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
                   cleanup_tmp: bool = True) -> dict:
        """
        执行完整的长音频转录流程。

        :param input_file: 要转录的媒体文件的本地路径或网络 URL。
        :param context: 提供给 ASR 模型的上下文提示词。
        :param save_outputs: 是否将结果保存到文件中。
        :param output_dir: 输出文件的存放目录。
        :param cleanup_tmp: 是否自动清理临时文件。
        :return: 包含转录结果的字典。
        """
        self._validate_input(input_file)

        tmp_save_dir = None
        try:
            # 1. 准备音频分段
            wav_list = self._prepare_audio_segments(input_file)

            # 2. 创建临时文件块
            wav_path_list, tmp_save_dir = self._create_tmp_chunks(wav_list, input_file)

            # 3. 并行执行转录
            raw_results, languages = self._run_parallel_transcription(wav_path_list, context)

            # 4. 合成最终结果
            final_results = self._compose_results(raw_results, languages, wav_list)

            # 5. 保存输出文件
            if save_outputs:
                saved_paths = self._save_results(final_results, input_file, output_dir)
                final_results.update(saved_paths)

            return final_results

        finally:
            # 6. 清理临时文件
            if cleanup_tmp and tmp_save_dir:
                self._cleanup(tmp_save_dir)


if __name__ == '__main__':
    #  创建 ASR 客户端实例
    #    确保您的 .env 文件或环境变量中设置了 DASHSCOPE_API_KEY
    qwen_client = QwenASR(model="qwen3-asr-flash")

    # 创建长音频转录器实例，注入 ASR 客户端
    transcriber = LongAudioTranscriber(
        asr_client=qwen_client,
        num_threads=4,  # 根据您的CPU核心数调整
        vad_segment_threshold_s = 180,
    )

    # 调用 transcribe 方法执行任务
    audio_file = r"D:\DATA\Voicer测试数据\如何学戒_五分钟.wav"
    result_data = transcriber.transcribe(
        input_file=audio_file,
        save_outputs=True,
    )
    print("\n--- Transcription Result ---")
    print(f"Language: {result_data['language']}")
    print(f"Full Text: {result_data['full_text'][:200]}...")  # 打印前200个字符预览
