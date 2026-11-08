import torch
from torch.utils.data import Dataset, DataLoader
from transformers import logging

import numpy as np
import ffmpeg
# import librosa
# import soundfile as sf
from pathlib import Path
import json
from typing import List, Optional, Any
from functools import lru_cache

from .schema import SingleSegment

logger = logging.get_logger(__name__)

DEFAULT_SAMPLE_RATE = 16000


@lru_cache(maxsize=1)
def _decode_raw_audio(path, target_sr, mtime_ns, size):
    """Raw AAC has no reliable seek index or header duration; decode it once."""
    out, _ = (ffmpeg.input(path).output("pipe:", format="f32le", ac=1, ar=target_sr)
              .run(capture_stdout=True, capture_stderr=True))
    return np.frombuffer(out, np.float32)

def read_audio(
    ele: dict,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    bounds_in_samples: bool = False,
):
    path = ele["audio"]

    # -------------------------
    # 1. 获取原始音频信息
    # -------------------------
    probe = ffmpeg.probe(path)

    audio_stream = next(
        s for s in probe["streams"]
        if s["codec_type"] == "audio"
    )

    orig_sr = int(audio_stream["sample_rate"])

    duration = float(
        probe["format"]["duration"]
    )
    decoded = None
    if probe["format"].get("format_name") == "aac":
        stat = Path(path).stat()
        decoded = _decode_raw_audio(str(path), target_sr, stat.st_mtime_ns, stat.st_size)
        duration = len(decoded) / target_sr

    # -------------------------
    # 2. 处理时间范围
    # -------------------------
    audio_start = ele.get("start", None)
    audio_end = ele.get("end", None)

    if bounds_in_samples:
        audio_start = (
            None
            if audio_start is None
            else float(audio_start) / target_sr
        )
        audio_end = (
            None
            if audio_end is None
            else float(audio_end) / target_sr
        )

    if audio_start is None:
        audio_start = 0.0

    if audio_end is None:
        audio_end = duration

    audio_start = max(0.0, float(audio_start))
    audio_end = min(float(audio_end), duration)

    if audio_end <= audio_start:
        raise ValueError(f"Invalid audio interval [{audio_start}, {audio_end}] for {path}")
        
    if decoded is not None:
        wav = decoded[round(audio_start * target_sr):round(audio_end * target_sr)]
    else:
        out, err = (
            ffmpeg
            .input(path, ss=audio_start, t=audio_end - audio_start)
            .filter("aresample", target_sr)
            .output("pipe:", format="f32le", ac=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
        wav = np.frombuffer(out, np.float32)

    # -------------------------
    # 4. clip pts
    # -------------------------
    nframes = len(wav)

    clip_pts = (
        audio_start
        + np.arange(nframes)
        / target_sr
    )


    wav = torch.from_numpy(
        wav.copy()
    ).float()


    return wav, clip_pts, target_sr


class AudioDatasetForSegment(Dataset):
    def __init__(self, audio_dir, target_sr=DEFAULT_SAMPLE_RATE):
        super().__init__()
        audio_dir = Path(audio_dir)
        
        extensions = {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".opus"}
        candidates = [audio_dir] if audio_dir.is_file() else audio_dir.rglob("*")
        self.audio_files = sorted(
            x for x in candidates if x.is_file() and x.suffix.lower() in extensions
        )
        if not self.audio_files:
            raise FileNotFoundError(f"No supported audio files found in {audio_dir}")
    
        self.target_sr = target_sr

    def __getitem__(self, index):
        audio_file = str(self.audio_files[index])
        waveform, _, sr = read_audio({"audio": audio_file}, target_sr=self.target_sr)
        return audio_file, waveform, sr
    
    def __len__(self):
        return len(self.audio_files)
    
    
def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        if end == 0:
            raise ValueError(f"Empty annotation file: {path}")
        pos = end - 1
        while pos >= 0:
            f.seek(pos)
            if f.read(1) == b"\n" and pos < end - 1:
                break
            pos -= 1
        f.seek(pos + 1)
        return f.readline().decode("utf-8")

    
class AudioDatasetForASR(Dataset):
    def __init__(self, annotation_dir: str=None, target_sr: int=16000, speech_only: bool=False):
        super().__init__()
        annotation_paths = [str(x) for x in sorted(Path(annotation_dir).glob("*.jsonl")) if ".rank" not in x.name]
        if not annotation_paths:
            raise FileNotFoundError(f"No merged annotation files found in {annotation_dir}")
        self.handles = []
        self.target_sr = target_sr
        for annotation_path in annotation_paths:
            assert annotation_path.endswith('.jsonl'), f"Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
            logger.warning(f'Load {annotation_path}. Please ensure its last line stores the seek indices...')
            seeks = json.loads(readlastline(annotation_path))
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            logger.warning(f'Successfully loaded {annotation_path}')
        if speech_only:
            self.handles = [handle for index, handle in enumerate(self.handles)
                            if self.load_segments(index)["type"] == "speech"]
            
    def load_segments(self, index):
        annotation_path, seek = self.handles[index]
        with open(annotation_path, "rb") as f:
            f.seek(seek)
            line = f.readline().decode("utf-8")
        line = json.loads(line)
        return line
    
    def getitem(self, index):
        segments = self.load_segments(index)
        audio_path = segments['audio_path']
        audio_type = segments['type']
        start, end = segments['start'], segments['end']
        # Legacy timestamp files were generated at 16 kHz. New files record it explicitly.
        timestamp_sr = segments.get("sampling_rate", DEFAULT_SAMPLE_RATE)
        audio = {"audio": audio_path, "start": start / timestamp_sr, "end": end / timestamp_sr}
        audio_clip, _, _ = read_audio(audio, self.target_sr)
        if len(audio_clip) == 0:
            raise ValueError(f"Empty speech segment: {audio}")
        return dict(
                    audio_path=audio_path, 
                    audio_type=audio_type,
                    waveform=audio_clip, 
                    sr=self.target_sr, 
                    start=start / timestamp_sr,
                    end=end / timestamp_sr
                )
    
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.handles)
    

class AudioDatasetForCaption(Dataset):
    def __init__(self, 
                 annotation_dir: str=None, 
                 target_sr: int=16000,
                 processor: Optional[Any] = None):
        super().__init__()
        annotation_paths = [str(x) for x in sorted(Path(annotation_dir).glob("*.jsonl"))]
        self.handles = []
        self.target_sr = target_sr
        if processor is None:
            raise ValueError("processor must be provided for AudioDatasetForCaption")
        self.processor = processor
        for annotation_path in annotation_paths:
            assert annotation_path.endswith('.jsonl'), f"Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
            logger.warning(f'Load {annotation_path}. Please ensure its last line stores the seek indices...')
            seeks = json.loads(readlastline(annotation_path))
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            logger.warning(f'Successfully loaded {annotation_path}')
            
    def load_segments(self, index):
        annotation_path, seek = self.handles[index]
        with open(annotation_path, "rb") as f:
            f.seek(seek)
            line = f.readline().decode("utf-8")
        line = json.loads(line)
        return line
    
    def get_prompt_text(self, segment_type: str) -> str:
        if segment_type == "environment_event":
            return "Identify the sound event and action. Short answer."
        elif segment_type == "background_ambience":
            return "Briefly describe the background scene and atmosphere. In one sentence."
        else:
            return "Describe this audio concisely."
    
    def getitem(self, index: int):
        segments = self.load_segments(index)
        audio_path = segments['audio_path']
        audio_type = segments['type']
        start, end = segments['start'], segments['end']
        audio = {"audio": audio_path, "start": start, "end": end}
        audio_clip, _, _ = read_audio(audio, self.target_sr, bounds_in_samples=True)

        prompt_content = self.get_prompt_text(audio_type)
        
        conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": "placeholder"},
                            {"type": "text", "text": prompt_content},
                        ],
                    },
                ]
        
        text = self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
        
        inputs = self.processor(
                    text=text,           # list of 1
                    audio=audio_clip,         # list of 1
                    return_tensors="pt",
                    use_audio_in_video=False
                )

        return inputs
    
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.handles)
    
LANGUAGES_MAP = {
    "Chinese": "zh",
    "English": "en",
}
    
class TranscriptDataset(Dataset):
    """
    Dataset for loading audio segments with their transcription from JSONL files.
    Each item corresponds to a single segment (one line) from a JSONL file.

    Args:
        annotation_dir (str): Path to the folder containing .jsonl files.
        target_sr (int): Target sample rate for audio resampling. Default: 16000.

    Each item returns:
        - waveform (Tensor): Audio waveform for the segment, shape [samples]
        - audio_path (str): The audio file path string from the JSONL
        - segment (SingleSegment): A single segment with keys 'text', 'start', 'end'
    """

    def __init__(self, annotation_dir: str, target_sr: int = 16000):
        self.folder_path = Path(annotation_dir)
        self.target_sr = target_sr
        self.items = []  # each item: (audio_path, SingleSegment)

        self._load_jsonl_files()

    def _load_jsonl_files(self):
        jsonl_files = sorted(self.folder_path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {self.folder_path}")

        for jsonl_file in jsonl_files:
            audio_path = None

            with open(jsonl_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[:-1]:  # skip last line (seek indices)
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if audio_path is None:
                        audio_path = record["audio_path"]

                    segment = SingleSegment(
                        text=record["text"],
                        start=record["start"],
                        end=record["end"]
                    )
                    
                    language = LANGUAGES_MAP.get(record["language"], record["language"])
                    
                    self.items.append((audio_path, segment, language))

    def __getitem__(self, idx):
        audio_path, segment, language = self.items[idx]

        start_sec = segment["start"]
        end_sec = segment["end"]

        waveform, _, _ = read_audio({"audio": audio_path}, self.target_sr)

        return waveform, audio_path, segment, language

    def __len__(self):
        return len(self.items)


def test_segment_dataset():
    from collator import DataCollatorForAudio
    collator = DataCollatorForAudio()
    dataset = AudioDatasetForSegment(audio_dir="/mnt/nvme/workspaces/liuyutong/vlog测试")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    for i, data in enumerate(dataloader):
        breakpoint()
        print(data["audio_path"])
        
def test_asr_dataset():
    from collator import DataCollatorForASR
    collator = DataCollatorForASR()
    dataset = AudioDatasetForASR(annotation_dir="../results/timestamps")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    for i, data in enumerate(dataloader):
        print(data["waveform"])
        
def test_caption_dataset():
    from collator import DataCollatorForCaption
    collator = DataCollatorForCaption()
    dataset = AudioDatasetForCaption(annotation_dir="/home/muyun/project/Annotation/results/timestamps")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    for i, data in enumerate(dataloader):
        breakpoint()
        print(data)
    
def test_transcript_dataset():
    from collator import DataCollatorForTranscripts
    collator = DataCollatorForTranscripts()
    dataset = TranscriptDataset(annotation_dir="/home/muyun/project/Annotation/results/transcripts")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    for i, data in enumerate(dataloader):
        breakpoint()
        print(data)
    

if __name__ == "__main__":
    test_transcript_dataset()
