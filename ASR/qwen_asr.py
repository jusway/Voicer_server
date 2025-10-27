import os
import time
import random
import dashscope
from pydub import AudioSegment
from ASR.base_asr import BaseASR

MAX_API_RETRY = 3
API_RETRY_SLEEP = (1, 2)


language_code_mapping = {
    "ar": "Arabic",
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish"
}


class QwenASR(BaseASR):
    def __init__(self, model: str = "qwen3-asr-flash"):
        self.model = model

    @property
    def max_duration_seconds(self) -> int:
        # 千问 API 的批量限制是 3 分钟
        return 180

    def post_text_process(self, text, threshold=20):
        def fix_char_repeats(s, thresh):
            res = []
            i = 0
            n = len(s)
            while i < n:
                count = 1
                while i + count < n and s[i + count] == s[i]:
                    count += 1

                if count > thresh:
                    res.append(s[i])
                    i += count
                else:
                    res.append(s[i:i + count])
                    i += count
            return ''.join(res)

        def fix_pattern_repeats(s, thresh, max_len=20):
            n = len(s)
            min_repeat_chars = thresh * 2
            if n < min_repeat_chars:
                return s

            i = 0
            result = []
            while i <= n - min_repeat_chars:
                found = False
                for k in range(1, max_len + 1):
                    if i + k * thresh > n:
                        break

                    pattern = s[i:i + k]

                    valid = True
                    for rep in range(1, thresh):
                        start_idx = i + rep * k
                        if s[start_idx:start_idx + k] != pattern:
                            valid = False
                            break

                    if valid:
                        total_rep = thresh
                        end_index = i + thresh * k
                        while end_index + k <= n and s[end_index:end_index + k] == pattern:
                            total_rep += 1
                            end_index += k

                        result.append(pattern)
                        result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                        i = n
                        found = True
                        break

                if found:
                    break
                else:
                    result.append(s[i])
                    i += 1

            if not found:
                result.append(s[i:])
            return ''.join(result)

        text = fix_char_repeats(text, threshold)
        return fix_pattern_repeats(text, threshold)

    def transcribe(self, wav_url: str, context: str = ""):
        if not wav_url.startswith("http"):
            assert os.path.exists(wav_url), f"{wav_url} not exists!"
            file_path = wav_url
            file_size = os.path.getsize(file_path)

            # file size > 10M
            if file_size > 10 * 1024 * 1024:
                # convert to mp3
                mp3_path = os.path.splitext(file_path)[0] + ".mp3"
                audio = AudioSegment.from_file(file_path)
                audio.export(mp3_path, format="mp3")
                wav_url = mp3_path

            wav_url = f"file://{wav_url}"

        # Submit the ASR task
        for _ in range(MAX_API_RETRY):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"text": context},
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"audio": wav_url},
                        ]
                    }
                ]
                response = dashscope.MultiModalConversation.call(
                    model=self.model,
                    messages=messages,
                    result_format="message",
                    asr_options={
                        "enable_lid": True,
                        "enable_itn": False
                    }
                )

                if response.status_code != 200:
                    raise Exception(f"http status_code: {response.status_code} {response}")
                output = response['output']['choices'][0]

                recog_text = None
                if len(output["message"]["content"]):
                    recog_text = output["message"]["content"][0]["text"]
                if recog_text is None:
                    recog_text = ""

                lang_code = None
                if "annotations" in output["message"]:
                    lang_code = output["message"]["annotations"][0]["language"]
                language = language_code_mapping.get(lang_code, "Not Supported")

                return language, self.post_text_process(recog_text)
            except Exception as e:
                print(f"Retry {_ + 1}...  {wav_url}\nError: {e}")
            time.sleep(random.uniform(*API_RETRY_SLEEP))
        raise Exception(f"{wav_url} task failed after {MAX_API_RETRY} retries. Last error: {e}")

if __name__ == "__main__":
    qwen_asr = QwenASR(model="qwen3-asr-flash")
    # asr_text = qwen_asr.asr(wav_url=r"D:\DATA\Voicer测试数据\如何学戒_五分钟.wav")
    asr_text = qwen_asr.transcribe(wav_url=r"D:\DATA\Voicer测试数据\如何学戒_0.5分钟.wav")

    print(asr_text)