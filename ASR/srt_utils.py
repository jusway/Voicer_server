
import os
from urllib.parse import urlparse

def format_srt_time(seconds: float) -> str:
    """将秒数转换为 SRT 格式时间字符串。

    参数:
    - seconds: 以秒为单位的时间，支持小数。

    返回:
    - 格式为 `HH:MM:SS,mmm` 的字符串。
    """
    total_ms = int(round(seconds * 1000))
    hours = total_ms // (3600 * 1000)
    minutes = (total_ms % (3600 * 1000)) // (60 * 1000)
    secs = (total_ms % (60 * 1000)) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def build_srt_entries(wav_list: list, raw_results: list, sample_rate: int) -> list:
    """根据分段样本范围与识别结果构造 SRT 条目列表。

    参数:
    - wav_list: 每段的起止样本及波形数据列表，元素为 `(start_sample, end_sample, wav)`。
    - raw_results: 识别结果列表，元素为 `(idx, text)`，函数内部按 `idx` 排序。
    - sample_rate: 音频采样率，用于将样本数转换为秒。

    返回:
    - SRT 条目字典列表，每项包含 `index`、`start`、`end`、`text`。

    说明:
    - 若 `raw_results` 项少于 `wav_list` 段数，对应文本将置为空字符串。
    """
    raw_results.sort(key=lambda x: x[0])
    srt_entries = []
    for idx, (start_sample, end_sample, _) in enumerate(wav_list):
        text = raw_results[idx][1] if idx < len(raw_results) else ""
        start_sec = start_sample / sample_rate
        end_sec = end_sample / sample_rate
        srt_entries.append({
            "index": idx + 1,
            "start": format_srt_time(start_sec),
            "end": format_srt_time(end_sec),
            "text": text.strip()
        })
    return srt_entries

def save_srt(srt_entries: list, input_file: str, output_dir: str) -> dict:
    """将构造好的 SRT 条目保存为 `.srt` 文件。

    参数:
    - srt_entries: 由 `build_srt_entries` 生成的条目列表。
    - input_file: 原始输入文件路径或 URL，用于确定输出文件名。
    - output_dir: 输出目录；为空时默认与输入文件同目录或当前目录。

    返回:
    - 包含键 `srt_file_path` 的字典，值为生成的 SRT 文件完整路径。

    副作用:
    - 在 `output_dir` 下创建并写入一个 `.srt` 文件。
    """
    if os.path.exists(input_file):
        base_name = os.path.splitext(os.path.basename(input_file))[0]
    else:
        base_name = os.path.splitext(urlparse(input_file).path)[0].split('/')[-1]

    output_dir = output_dir or os.path.dirname(input_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    srt_file_path = os.path.join(output_dir, f"{base_name}.srt")

    with open(srt_file_path, 'w', encoding='utf-8') as f:
        for item in srt_entries:
            f.write(f"{item['index']}\n")
            f.write(f"{item['start']} --> {item['end']}\n")
            f.write(f"{item['text']}\n\n")
    return {"srt_file_path": srt_file_path}