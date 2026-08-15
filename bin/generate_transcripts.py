import argparse
from pathlib import Path
import os
import torch
import json

from audio.asr_model import WhisperASR, QwenASR
from data.dataset import AudioDatasetForASR
from data.collator import DataCollatorForASR
from data.utils import extract_bv_id_from_dir

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--asr-model-dir",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )
    
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000
    )
    
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="./results/timestamps"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/transcripts"
    )
            
class AnnotationAudio:
    def __init__(self, 
                 annotation_dir: str,
                 batch_size: int=4, 
                 target_sr: int=16000, 
                 asr_model_dir: str=None, 
                 output_dir: str="./outputs",
                 local_rank: int = 0,
                 world_size: int = 1,
            ):
        
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        
        torch.cuda.set_device(self.device)
        
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_sr = target_sr
        if "whisper" in asr_model_dir.lower():
            self.asr_model = WhisperASR(
                model_name=asr_model_dir,
                language="ja",
                device=self.device,
            )
            
        elif "qwen3" in asr_model_dir.lower():
            self.asr_model = QwenASR(
                model_name=asr_model_dir,
                device=self.device,
            )   
            
        segments_dataset = AudioDatasetForASR(annotation_dir=annotation_dir, target_sr=target_sr)
        
        rank_dataset = torch.utils.data.Subset(
            segments_dataset,
            range(local_rank, len(segments_dataset), world_size),
        )
        
        self.segments_loader = torch.utils.data.DataLoader(
            rank_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=DataCollatorForASR(),
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
            f.write(json.dumps(seeks, ensure_ascii=True))
            f.close()
            f = None

        for batch in self.segments_loader:
            types = batch["type"]
            is_speech = torch.tensor([t == "speech" for t in types], dtype=torch.bool)
            if not is_speech.any():
                continue
            waveforms = batch["waveforms"][is_speech]
            srs = batch["sampling_rate"][is_speech]
            starts = batch["start"][is_speech]
            ends = batch["end"][is_speech]
            audio_paths = [batch["audio_path"][i] for i in range(len(is_speech)) if is_speech[i]]
            masks = batch["attention_mask"][is_speech]
            inputs = [
                (w[: int(mask.sum())].cpu().numpy(), int(sr.item()))
                for w, mask, sr in zip(waveforms, masks, srs)
            ]
            texts, languages = self.asr_model.transcribe(
                audio=inputs
            )
            for path, start, end, text, lang in zip(
                audio_paths, starts, ends, texts, languages
            ):
                key = f"{extract_bv_id_from_dir(Path(path).parent.stem)}_{Path(path).stem}.rank{self.local_rank}"

                if key != current_key:
                    close_file()
                    current_key = key
                    lengths = []

                    out_file = self.output_dir / f"{key}.jsonl"
                    f = open(out_file, "w", encoding="utf-8")

                line = json.dumps(
                    {
                        "audio_path": path,
                        "start": start.item() / self.target_sr,
                        "end": end.item() / self.target_sr,
                        "text": text.strip(),
                        "language": lang.strip(),
                    },
                    ensure_ascii=False,
                ) + "\n"

                lengths.append(len(line.encode("utf-8")))
                f.write(line)

        close_file()
                    
            
            
def run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()
    
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    annotator = AnnotationAudio(
        annotation_dir=args.annotation_dir,
        batch_size=args.batch_size,
        target_sr=args.target_sr,
        asr_model_dir=args.asr_model_dir,
        output_dir=args.output_dir,
        local_rank=rank,
        world_size=world_size
    )
    annotator.generate()
    
if __name__ == "__main__":
    run()
