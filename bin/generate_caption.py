import argparse
from pathlib import Path
import os
import torch
import json

from audio.sed_model import Qwen3OmniCaptioner
from data.dataset import AudioDatasetForASR
from data.collator import DataCollatorForASR
from data.utils import extract_bv_id_from_dir

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--sed-model-dir",
        type=str,
        default="/mnt/nvme/juchen/qwen/Qwen3-Omni-30B-A3B-Instruct",
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
        default="./results/caption"
    )
            
class AnnotationAudio:
    def __init__(self, 
                 annotation_dir: str,
                 batch_size: int=4, 
                 target_sr: int=16000, 
                 sed_model_dir: str=None, 
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
        
        self.sed_model = Qwen3OmniCaptioner(
                model_name=sed_model_dir,
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
            num_workers=4,
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
            f.write(json.dumps(seeks, ensure_ascii=False))
            f.close()
            f = None

        for batch in self.segments_loader:
            types = batch["type"]
            is_speech = torch.tensor([t == "speech" for t in types], dtype=torch.bool)
            is_env = ~is_speech
            if not is_env.any():
                continue
            env_types = [t for t, is_s in zip(types, is_speech.tolist()) if not is_s]
            waveforms = batch["waveforms"][is_env]
            masks = batch["attention_mask"][is_env]
            starts = batch["start"][is_env]
            ends = batch["end"][is_env]
            audio_paths = [batch["audio_path"][i] for i in range(len(is_env)) if is_env[i]]
            
            unpadded_waveforms = [
                waveform[: int(mask.sum())].cpu().numpy()
                for waveform, mask in zip(waveforms, masks)
            ]
            texts = self.sed_model.generate(unpadded_waveforms, env_types)

            for path, start, end, text, env_type in zip(
                audio_paths, starts, ends, texts, env_types
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
                        "type": env_type,
                        "start": start.item() / self.target_sr,
                        "end": end.item() / self.target_sr,
                        "text": text.strip(),
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
        sed_model_dir=args.sed_model_dir,
        output_dir=args.output_dir,
        local_rank=rank,
        world_size=world_size
    )
    annotator.generate()
    
if __name__ == "__main__":
    run()
