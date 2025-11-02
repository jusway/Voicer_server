import os
import requests
import concurrent.futures
import shutil
from collections import Counter
import gradio as gr

from ASR.base_asr import BaseASR
from ASR.audio_tools import load_audio, process_vad, save_audio_file, WAV_SAMPLE_RATE
from silero_vad import load_silero_vad
from ASR.srt_utils import build_srt_entries, save_srt


class SubtitleOrchestrator:
    """
    长音频/视频 → SRT 字幕的编排器。

    责任：
    - 加载音频/视频
    - 采用“字幕友好”的细粒度分段（目标约 4 秒，范围 2–5 秒）
    - 并行调用 ASR 客户端完成转写
    - 生成并保存 SRT 文件
    """

    def __init__(self,
                 asr_client: BaseASR,
                 num_threads: int = 4,
                 tmp_dir: str = os.path.join(os.path.expanduser("~"), "qwen3-srt-cache")):
        if not isinstance(asr_client, BaseASR):
            raise TypeError(f"asr_client must be an instance of BaseASR, but got {type(asr_client)}")

        self.asr_client = asr_client
        self.num_threads = num_threads
        self.tmp_dir = tmp_dir
        self.vad_model = None

    def _load_vad_model(self):
        """延迟加载 Silero VAD 模型以节省资源。"""
        if self.vad_model is None:
            print("Initializing Silero VAD model for subtitle segmenting...")
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

    def _prepare_subtitle_segments(self, file_path: str,
                                   target_s: int = 4, min_s: int = 2, max_s: int = 5) -> list:
        """
        生成适合字幕（2–5秒）的音频分段：
        - 先用 VAD 做细粒度分段（目标约 4 秒，最大不超过 5 秒）
        - 再对结果进行二次约束：合并过短片段，切分过长片段
        - VAD 失败时退化为固定时窗切分
        """
        wav = load_audio(file_path)
        self._load_vad_model()

        # 先尝试 VAD 细粒度分段
        try:
            base_segments = process_vad(
                wav,
                self.vad_model,
                segment_threshold_s=target_s,
                max_segment_threshold_s=max_s
            )
        except Exception:
            base_segments = []

        # 退化：固定窗口切分（max_s 秒）
        if not base_segments:
            segmented_wavs = []
            total_samples = len(wav)
            chunk_samples = int(max_s * WAV_SAMPLE_RATE)
            start = 0
            while start < total_samples:
                end = min(start + chunk_samples, total_samples)
                if end - start > 0:
                    segmented_wavs.append((start, end, wav[start:end]))
                start = end
            return segmented_wavs

        # 二次约束：合并过短、切分过长
        result = []
        buffer_start = None
        buffer_end = None
        max_samples = int(max_s * WAV_SAMPLE_RATE)
        min_samples = int(min_s * WAV_SAMPLE_RATE)

        def flush_buffer(start, end):
            # 将 [start, end] 按 <= max_s 切分并输出到 result
            cur = start
            while (end - cur) > max_samples:
                piece_end = cur + max_samples
                result.append((cur, piece_end, wav[cur:piece_end]))
                cur = piece_end
            result.append((cur, end, wav[cur:end]))

        for (seg_start, seg_end, _) in base_segments:
            if buffer_start is None:
                buffer_start, buffer_end = seg_start, seg_end
                continue

            # 若缓冲段不足 min_s，则合并下一个分段
            if (buffer_end - buffer_start) < min_samples:
                buffer_end = seg_end
            else:
                flush_buffer(buffer_start, buffer_end)
                buffer_start, buffer_end = seg_start, seg_end

        # 刷新剩余缓冲
        if buffer_start is not None:
            # 如果最后一段太短，尝试与前一段合并并重新切分
            if (buffer_end - buffer_start) < min_samples and len(result) > 0:
                prev_start, prev_end, _ = result.pop()
                merged_start, merged_end = prev_start, buffer_end
                flush_buffer(merged_start, merged_end)
            else:
                flush_buffer(buffer_start, buffer_end)

        # 最后一轮微调：避免出现 < min_s 的尾段
        if len(result) >= 2:
            last_start, last_end, _ = result[-1]
            if (last_end - last_start) < min_samples:
                # 合并到前一段，再按 max_s 切分
                prev_start, prev_end, _ = result.pop()
                merged_start, merged_end = prev_start, last_end
                # 使用相同的 flush 逻辑
                cur = merged_start
                while (merged_end - cur) > max_samples:
                    piece_end = cur + max_samples
                    result.append((cur, piece_end, wav[cur:piece_end]))
                    cur = piece_end
                result.append((cur, merged_end, wav[cur:merged_end]))

        return result

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
            # 简化：不引入 tqdm 的进度条，这里由 gr.Progress 外层展示即可
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
        return results, languages

    def generate_srt(self,
                     input_file: str,
                     context: str = "",
                     output_dir: str = None,
                     cleanup_tmp: bool = True,
                     progress: gr.Progress = None) -> dict:
        """
        执行完整的字幕生成流程：字幕分段（2–5秒）→ 并行调用 ASR → 生成并保存 SRT。
        返回：{"language": str, "srt_file_path": str}
        """
        self._validate_input(input_file)

        tmp_save_dir = None
        try:
            if progress:
                progress(0.1, desc="音频准备中...字幕分段策略(2–5秒)...")
            wav_list = self._prepare_subtitle_segments(input_file, target_s=4, min_s=2, max_s=5)

            if progress:
                progress(0.2, desc="分段处理中...")
            wav_path_list, tmp_save_dir = self._create_tmp_chunks(wav_list, input_file)

            if progress:
                progress(0.3, desc="收到任务，开始调用ASR（细粒度字幕）...")
            raw_results, languages = self._run_parallel_transcription(wav_path_list, context)

            if progress:
                progress(0.95, desc="生成 SRT...")
            srt_entries = build_srt_entries(wav_list, raw_results, WAV_SAMPLE_RATE)

            language = Counter(languages).most_common(1)[0][0] if languages else "Unknown"
            saved_paths = save_srt(srt_entries, input_file, output_dir)

            if progress:
                progress(0.99, desc="完成")
            return {
                "language": language,
                **saved_paths
            }
        finally:
            if cleanup_tmp and tmp_save_dir:
                if os.path.exists(tmp_save_dir):
                    shutil.rmtree(tmp_save_dir)
                    print(f"Cleaned up temporary directory: {tmp_save_dir}")


if __name__ == '__main__':
    # --- 使用示例 ---
    from ASR.qwen_asr import QwenASR
    import config

    qwen_client = QwenASR(model="qwen3-asr-flash")
    orchestrator = SubtitleOrchestrator(
        asr_client=qwen_client,
        num_threads=4,
    )
    orchestrator.generate_srt(input_file=r"D:\DATA\Voicer测试数据\如何学戒_五分钟.wav")