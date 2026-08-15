from audio.vad_model import VAD
from audio.env_model import EnvDetect
from data.dataset import AudioDatasetForSegment
from data.collator import DataCollatorForAudio
from data.utils import extract_bv_id_from_dir

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
    batch_size: int= 4
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

        self.audio_dataset = AudioDatasetForSegment(
            audio_dir=audio_dir,
            target_sr=sampling_rate,
        )
        
        rank_dataset = torch.utils.data.Subset(
            self.audio_dataset,
            range(rank, len(self.audio_dataset), world_size),
        )
        
        self.audio_loader = torch.utils.data.DataLoader(
            rank_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
            collate_fn=DataCollatorForAudio(),
            pin_memory=True,
        )
        
    def generate(self):
        current_key = None
        f = None
        lengths = []
        
        def close_file():
            nonlocal f, lengths, current_key
            if f is None:
                return
            
            seeks = (
                [0] + torch.tensor(lengths, dtype=torch.long).cumsum(0).tolist()[:-1]
                if lengths else []
            )
            
            print(f"Finished writing {current_key}, total segments: {len(lengths)}")
            f.write(json.dumps(seeks, ensure_ascii=False))
            f.close()
            f = None
        
        for batch in self.audio_loader:
            audio_paths = batch["audio_path"]
            waveforms = batch["input_values"]
            attention_masks = batch["attention_mask"]
            srs = batch["sampling_rate"]
            
            for audio_path, sr, waveform, attention_mask in zip(audio_paths, srs, waveforms, attention_masks):
                index = torch.sum(attention_mask, dim=0).item()
                waveform = waveform[:index]
                speech_segment = self.vad_model.generate(
                    waveform=waveform,
                    sampling_rate=self.sampling_rate,
                    threshold=self.threshold,
                    min_silence_duration_ms=self.min_silence_duration_ms,
                    min_speech_duration_ms=self.min_speech_duration_ms
                )

                
                env_segments, filtered_bg_segments = self.env_model.generate(waveform, speech_segment)

                all_results = []
                for s, e in speech_segment:
                    all_results.append({"type": "speech", "audio_path": audio_path, "start": s, "end": e})
                for s, e in env_segments:
                    all_results.append({"type": "environment_event", "audio_path": audio_path, "start": s, "end": e})
                for s, e in filtered_bg_segments:
                    all_results.append({"type": "background_ambience", "audio_path": audio_path, "start": s, "end": e})
                
                all_results.sort(key=lambda x: x['start'])
                
                key = f"{extract_bv_id_from_dir(Path(audio_path).parent.stem)}_{Path(audio_path).stem}.rank{self.local_rank}"

                if key != current_key:
                    close_file()
                    current_key = key
                    lengths = []

                    out_file = self.output_dir / f"{key}.jsonl"
                    f = open(out_file, "w", encoding="utf-8")
                
                for results in all_results:
                    line = json.dumps(results, ensure_ascii=True) + "\n"
                    lengths.append(len(line.encode("utf-8")))
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
        batch_size=data_args.batch_size,
        sampling_rate=data_args.sampling_rate,
        threshold=model_args.threshold,
        min_silence_duration_ms=model_args.min_silence_duration_ms,
        min_speech_duration_ms=model_args.min_speech_duration_ms
    )
    segmentor.generate()
    
if __name__ == "__main__":
    run()
