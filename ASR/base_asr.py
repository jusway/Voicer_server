# ASR/base_asr.py
from abc import ABC, abstractmethod


class BaseASR(ABC):

    @property
    @abstractmethod
    def max_duration_seconds(self) -> int:
        """
        返回 ASR API 单次请求能处理的
        最大音频时长（秒）。
        """
        pass

    @abstractmethod
    def transcribe(self, audio_file_path: str, context: str) -> tuple[str, str]:
        """
        转录一个 *单个* 音频文件。

        :param audio_file_path: 音频文件的路径 (str)。
        :param context: 上下文提示词 (str)。
        :return: 一个元组 (language: str, text: str)。
        """
        pass