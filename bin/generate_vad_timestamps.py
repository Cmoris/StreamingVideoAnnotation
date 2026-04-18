from audio import *
from data.dataset import AudioDatasetForsegment
from data.collator import DataCollatorForAudio

import torch
from transformers import HfArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
import os

@dataclass
class DataArguments:
    audio_dir: str = "./sources"
    output_dir: str = "./outputs"
    sampling_rate: int = 16000
    
@dataclass
class ModelArguments:
    vad_model_dir: str = None
    threshold: float=0.5
    min_silence_duration_ms: int=250
    min_speech_duration_ms: int=500
    
class VADSegment:
    def __init__(
        self,
        model_dir: str,
        audio_dir: str,
        output_dir: str,
        batch_size: int=4,
        sampling_rate: int = 16000,
        threshold=0.5,
        min_silence_duration_ms=250,
        min_speech_duration_ms=500,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.local_rank = rank
        self.world_size = world_size
        self.device = f"cuda:{rank}"

        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms

        torch.cuda.set_device(rank)

        self.vad_model = VAD(
            model_dir=model_dir,
            device=self.device,
        )
        
        self.env_model = EnvDetect(
            sampling_rate=sampling_rate,
            device=self.device
        )

        self.audio_dataset = AudioDatasetForsegment(
            audio_dir=audio_dir,
            target_sr=sampling_rate,
        )
        
        sampler = torch.utils.data.DistributedSampler(
            self.audio_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        
        self.audio_loader = torch.utils.data.DataLoader(
            self.audio_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            collate_fn=DataCollatorForAudio(),
            pin_memory=True,
        )
        
    def generate(self):
        current_key = None
        f = None
        lengths = []
        
        def close_file():
            if f is None:
                return
            seeks = [0] + torch.tensor(lengths).cumsum(0).tolist()[:-1]
            f.write(json.dumps(seeks, ensure_ascii=False))
            f.close()
        
        for batch in self.audio_loader:
            audio_paths = batch["audio_path"]
            waveforms = batch["input_values"]
            attention_masks = batch["attention_mask"]
            srs = batch["sampling_rate"]
            
            for audio_path, sr, waveform, attention_mask in zip(audio_paths, srs, waveforms, attention_masks):
                index = torch.cumsum(attention_mask.all())
                waveform = waveform[:index]
                speech_segment = self.vad_model.generate(
                    waveform=waveform,
                    sampling_rate=self.sampling_rate,
                    threshold=self.threshold,
                    min_silence_duration_ms=self.min_silence_duration_ms,
                    min_speech_duration_ms=self.min_speech_duration_ms
                )
                
                for results in speech_segment:
                
                    key = f"{Path(audio_path).stem}.rank{self.local_rank}"

                    if key != current_key:
                        close_file()
                        current_key = key
                        lengths = []

                        out_file = self.output_dir / f"{key}.jsonl"
                        f = open(out_file, "w", encoding="utf-8")
                    
                    line = json.dumps(results, ensure_ascii=True) + "\n"
                    
                    lengths.append(len(line))
                    f.write(line)

            close_file()

            
def run():
    model_args, data_args = HfArgumentParser((ModelArguments, DataArguments)).parse_args_into_dataclasses()
    
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    segmentor = VADSegment(
        rank=rank,
        world_size=world_size,
        model_dir=model_args.vad_model_dir,
        audio_dir=data_args.audio_dir,
        output_dir=data_args.output_dir,
        sampling_rate=data_args.sampling_rate,
        threshold=model_args.threshold,
        min_silence_duration_ms=model_args.min_silence_duration_ms,
        min_speech_duration_ms=model_args.min_speech_duration_ms
    )
    segmentor.generate()
    
if __name__ == "__main__":
    run()