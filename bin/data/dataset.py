from torch.utils.data import Dataset, DataLoader
from transformers import logging, AutoProcessor
from torchcodec.decoders import AudioDecoder

import numpy as np
import librosa
from pathlib import Path
import json
from typing import List, Dict

from .schema import SingleSegment
from .utils import collect_all_best_audios, extract_audio_from_video

logger = logging.get_logger(__name__)

def read_audio(ele: dict, sampling_rate: int):
    audio_decoder = AudioDecoder(source=ele['Audio Path'], sample_rate=sampling_rate)
    audio_sr = audio_decoder.metadata.sample_rate
    audio_duration = audio_decoder.metadata.duration_seconds_from_header
    total_frames = int(audio_duration*audio_sr)
    audio_pts = np.linspace(1/audio_sr, audio_duration, total_frames)
    audio_start = ele.get("start", None)
    audio_end = ele.get("end", None)
    clip_idxs = None
    if audio_start is not None or audio_end is not None:
        audio_start = audio_pts[0] if not audio_start else audio_start
        audio_end = audio_pts[-1] if not audio_end else audio_end
        clip_idxs = ((audio_start <= audio_pts) & (audio_pts <= audio_end)).nonzero()[0]
        clip_pts = audio_pts[clip_idxs]
        total_frames = len(clip_pts)
    else:
        audio_start = 0
        audio_end = audio_duration
        
    nframes = int(total_frames/audio_sr*sampling_rate)
    nframes_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    clip_idxs = nframes_idxs if clip_idxs is None else clip_idxs[nframes_idxs]
    clip_pts = audio_pts[clip_idxs]
    clip = audio_decoder.get_samples_played_in_range(start_seconds=audio_start, stop_seconds=audio_end+1/sampling_rate).data.squeeze(0)
    
    return clip, clip_pts, audio_sr

class VideoDatasetForSegment(Dataset):
    def __init__(self, dir, target_sr=16000):
        super().__init__()
        self.video_files = list(Path(dir).glob("*.mp4"))
        self.target_sr = target_sr
        
    def __getitem__(self, index):
        video_file = self.video_files[index]
        audio_clip = extract_audio_from_video(video_file, sample_rate=self.target_sr)
        return str(video_file), audio_clip, self.target_sr

class AudioDatasetForsegment(Dataset):
    def __init__(self, audio_dir, target_sr=16000):
        super().__init__()
        audio_dir = Path(audio_dir)
        
        self.audio_files = collect_all_best_audios(audio_dir)
    
        self.target_sr = target_sr

    def __getitem__(self, index):
        audio_file = str(self.audio_files[index])
        waveform, sr = librosa.load(audio_file, sr=self.target_sr, mono=True)
        return audio_file, waveform, sr
    
    def __len__(self):
        return len(self.audio_files)
    
    
def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2) # avoid last \n
        while f.read(1) != b"\n":  
            f.seek(-2, 1)
        return f.readline()

    
class AudioDatasetForASR(Dataset):
    def __init__(self, annotation_dir: str=None, target_sr: int=1600):
        super().__init__()
        annotation_paths = [str(x) for x in Path(annotation_dir).glob("*.jsonl")]
        self.handles = []
        self.target_sr = target_sr
        for annotation_path in annotation_paths:
            assert annotation_path.endswith('.jsonl'), f"Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
            logger.warning(f'Load {annotation_path}. Please ensure its last line stores the seek indices...')
            seeks = json.loads(readlastline(annotation_path))
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            logger.warning(f'Successfully loaded {annotation_path}')
            
    def load_segments(self, index):
        annotation_path, seek = self.handles[index]
        with open(annotation_path) as f:
            f.seek(seek)
            line = f.readline()
        line = json.loads(line)
        return line
    
    def getitem(self, index):
        segments = self.load_segments(index)
        audio_path = segments['Audio Path']
        audio_type = segments['Type']
        start, end = segments['Start'], segments['End']
        audio_decoder = AudioDecoder(audio_path, sample_rate=self.target_sr, num_channels=1)
        audio_clip = audio_decoder.get_samples_played_in_range(start_seconds=start/16000, stop_seconds=(end+1)/16000).data.squeeze(0)

        return dict(
                    audio_path=audio_path, 
                    audio_type=audio_type,
                    waveform=audio_clip, 
                    sr=self.target_sr, 
                    start=start, 
                    end=end
                )
    
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.handles)
    

class AudioDatasetForASRSingle(Dataset):
    def __init__(self, annotation_path: str=None, target_sr: int=16000):
        super().__init__()
        self.handles = []
        self.target_sr = target_sr
        assert annotation_path.endswith('.jsonl'), f"Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
        logger.warning(f'Load {annotation_path}. Please ensure its last line stores the seek indices...')
        seeks = json.loads(readlastline(annotation_path))
        self.handles.extend(zip([annotation_path] * len(seeks), seeks))
        logger.warning(f'Successfully loaded {annotation_path}')
            
    def load_segments(self, index):
        annotation_path, seek = self.handles[index]
        with open(annotation_path) as f:
            f.seek(seek)
            line = f.readline()
        line = json.loads(line)
        return line
    
    def getitem(self, index):
        segments = self.load_segments(index)
        audio_path = segments['Audio Path']
        start, end = segments['Start'], segments['End']
        audio_decoder = AudioDecoder(audio_path, sample_rate=self.target_sr, num_channels=1)
        audio_clip = audio_decoder.get_samples_played_in_range(start_seconds=start/self.target_sr, stop_seconds=(end+1)/self.target_sr).data.squeeze(0)
            
        return dict(
                audio_path=audio_path, 
                waveform=audio_clip, 
                sr=self.target_sr, 
                start=start, 
                end=end
            )
    
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.handles)
    

class AudioDatasetForCaption(Dataset):
    def __init__(self, 
                 annotation_dir: str=None, 
                 target_sr: int=16000,
                 processor: AutoProcessor=AutoProcessor.from_pretrained("/mnt/nvme/juchen/qwen/Qwen3-Omni-30B-A3B-Instruct")):
        super().__init__()
        annotation_paths = [str(x) for x in Path(annotation_dir).glob("*.jsonl")]
        self.handles = []
        self.target_sr = target_sr
        self.processor = processor
        for annotation_path in annotation_paths:
            assert annotation_path.endswith('.jsonl'), f"Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
            logger.warning(f'Load {annotation_path}. Please ensure its last line stores the seek indices...')
            seeks = json.loads(readlastline(annotation_path))
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            logger.warning(f'Successfully loaded {annotation_path}')
            
    def load_segments(self, index):
        annotation_path, seek = self.handles[index]
        with open(annotation_path) as f:
            f.seek(seek)
            line = f.readline()
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
        audio_path = segments['Audio Path']
        audio_type = segments['Type']
        start, end = segments['Start'], segments['End']
        audio_decoder = AudioDecoder(audio_path, sample_rate=self.target_sr, num_channels=1)
        audio_clip = audio_decoder.get_samples_played_in_range(start_seconds=start/self.target_sr, stop_seconds=(end+1)/self.target_sr).data.squeeze(0)

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
    
class TranscriptDatasetAll(Dataset):
    """
    Dataset for loading audio files with their transcription segments from JSONL files.

    Args:
        folder_path (str): Path to the folder containing .jsonl files.
        sample_rate (int): Target sample rate for audio resampling. Default: 16000.

    Each item returns:
        - waveform (Tensor): Full audio waveform, shape [channels, samples]
        - audio_path (str): The audio file path string from the JSONL
        - segments (list): List of dicts with keys 'text', 'start', 'end'
    """

    def __init__(self, annotation_dir: str, target_sr: int = 16000):
        self.folder_path = Path(annotation_dir)
        self.target_sr = target_sr
        self.items = []  # each item: (audio_path, segments)

        self._load_jsonl_files()

    def _load_jsonl_files(self):
        jsonl_files = sorted(self.folder_path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {self.folder_path}")

        for jsonl_file in jsonl_files:
            audio_path = None
            segments :List[SingleSegment] = []

            with open(jsonl_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[:-1]: # skip the last line which stores the seek indices
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if audio_path is None:
                        audio_path = record["audio_path"]

                    segments.append(SingleSegment(
                        text=record["text"],
                        start=record["start"],
                        end=record["end"]
                    ))

            if audio_path is not None and segments:
                self.items.append((audio_path, segments))

    def __getitem__(self, idx):
        audio_path, segments = self.items[idx]

        audio_decoder = AudioDecoder(audio_path, sample_rate=self.target_sr, num_channels=1)
        waveform = audio_decoder.get_all_samples().data.squeeze(0)

        return waveform, audio_path, segments
    
    def __len__(self):
        return len(self.items)
    
    
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

        audio_decoder = AudioDecoder(
            audio_path,
            sample_rate=self.target_sr,
            num_channels=1
        )

        waveform = audio_decoder.get_all_samples().data.squeeze(0)

        return waveform, audio_path, segment, language

    def __len__(self):
        return len(self.items)


def test_segment_dataset():
    from collator import DataCollatorForAudio
    collator = DataCollatorForAudio()
    dataset = AudioDatasetForsegment(audio_dir="/mnt/nvme/workspaces/liuyutong/vlog测试")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    for i, data in enumerate(dataloader):
        breakpoint()
        print(data["audio_path"])
        
def test_asr_dataset():
    from collator import DataCollatorForASR
    collator = DataCollatorForASR()
    dataset = AudioDatasetForASR(annotation_dir="/home/muyun/project/Annotation/results/timestamps")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    for i, data in enumerate(dataloader):
        print(data)
        
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