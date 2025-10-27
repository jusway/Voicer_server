import os
import io
import librosa
import subprocess
import numpy as np
import soundfile as sf
from silero_vad import get_speech_timestamps

WAV_SAMPLE_RATE = 16000


def load_audio(file_path: str) -> np.ndarray:
    try:
        if file_path.startswith(("http://", "https://")):
            raise ValueError("Using ffmpeg to load remote file.")

        # Try librosa first, because it is usually faster for standard formats.
        wav_data, _ = librosa.load(file_path, sr=WAV_SAMPLE_RATE, mono=True)
        return wav_data
    except Exception as e:
        print(e)

        # After librosa fails, use a more powerful ffmpeg as a backup.
        try:
            command = [
                'ffmpeg',
                '-i', file_path,
                '-ar', str(WAV_SAMPLE_RATE),
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                '-f', 'wav',
                '-'
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout_data, stderr_data = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error processing local file: {stderr_data.decode('utf-8', errors='ignore')}")

            with io.BytesIO(stdout_data) as data_io:
                wav_data, sr = sf.read(data_io, dtype='float32')

            return wav_data
        except Exception as ffmpeg_e:
            raise RuntimeError(f"Failed to load audio from local file '{file_path}' even with ffmpeg. Error: {ffmpeg_e}")


def process_vad(wav: np.ndarray, worker_vad_model, segment_threshold_s: int = 120, max_segment_threshold_s: int = 180) -> list[np.ndarray]:
    """
    将这个长音频智能地切分成一系列 2-3 分钟的、在句子间断开的音频块。
    """
    try:
        vad_params = {
            'sampling_rate': WAV_SAMPLE_RATE,
            'return_seconds': False,
            'min_speech_duration_ms': 1500,
            'min_silence_duration_ms': 500
        }

        speech_timestamps = get_speech_timestamps(
            wav,
            worker_vad_model,
            **vad_params
        )

        if not speech_timestamps:
            raise ValueError("No speech segments detected by VAD.")

        potential_split_points_s = {0.0, len(wav)}
        for i in range(len(speech_timestamps)):
            start_of_next_s = speech_timestamps[i]['start']
            potential_split_points_s.add(start_of_next_s)
        sorted_potential_splits = sorted(list(potential_split_points_s))

        final_split_points_s = {0.0, len(wav)}
        segment_threshold_samples = segment_threshold_s * WAV_SAMPLE_RATE
        target_time = segment_threshold_samples
        while target_time < len(wav):
            closest_point = min(sorted_potential_splits, key=lambda p: abs(p - target_time))
            final_split_points_s.add(closest_point)
            target_time += segment_threshold_samples
        final_ordered_splits = sorted(list(final_split_points_s))

        max_segment_threshold_samples = max_segment_threshold_s * WAV_SAMPLE_RATE
        new_split_points = [0.0]

        # Make sure that each audio segment does not exceed max_segment_threshold_s
        for i in range(1, len(final_ordered_splits)):
            start = final_ordered_splits[i - 1]
            end = final_ordered_splits[i]
            segment_length = end - start

            if segment_length <= max_segment_threshold_samples:
                new_split_points.append(end)
            else:
                num_subsegments = int(np.ceil(segment_length / max_segment_threshold_samples))
                subsegment_length = segment_length / num_subsegments

                for j in range(1, num_subsegments):
                    split_point = start + j * subsegment_length
                    new_split_points.append(split_point)

                new_split_points.append(end)

        segmented_wavs = []
        for i in range(len(new_split_points) - 1):
            start_sample = int(new_split_points[i])
            end_sample = int(new_split_points[i + 1])
            segmented_wavs.append((start_sample, end_sample, wav[start_sample:end_sample]))
        return segmented_wavs

    except Exception as e:
        segmented_wavs = []
        total_samples = len(wav)
        max_chunk_size_samples = max_segment_threshold_s * WAV_SAMPLE_RATE

        for start_sample in range(0, total_samples, max_chunk_size_samples):
            end_sample = min(start_sample + max_chunk_size_samples, total_samples)
            segment = wav[start_sample:end_sample]
            if len(segment) > 0:
                segmented_wavs.append((start_sample, end_sample, segment))

        return segmented_wavs


def save_audio_file(wav: np.ndarray, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, wav, WAV_SAMPLE_RATE)