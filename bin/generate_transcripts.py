import argparse
from pathlib import Path
import os
import torch
import json

from audio.asr_model import WhisperASR, QwenASR
from data.dataset import AudioDatasetForASR, read_audio
from data.collator import DataCollatorForASR
from data.utils import extract_bv_id_from_dir

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--language", default=None, help="Language hint; omit for automatic detection")
    parser.add_argument("--asr-backend", choices=("whisper", "qwen3"), default=None)
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
                 rank: int = None,
                 language: str = None,
                 asr_backend: str = None,
                 num_workers: int = 0,
                 device: str = None,
            ):
        
        self.local_rank = local_rank if rank is None else rank
        self.world_size = world_size
        self.device = device or f"cuda:{local_rank}"
        
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
        
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_sr = target_sr
        backend = asr_backend or ("whisper" if "whisper" in asr_model_dir.lower() else "qwen3" if "qwen3" in asr_model_dir.lower() else None)
        if backend == "whisper":
            self.asr_model = WhisperASR(
                model_name=asr_model_dir,
                language=language,
                device=self.device,
            )
            
        elif backend == "qwen3":
            self.asr_model = QwenASR(
                model_name=asr_model_dir,
                device=self.device,
                language=language,
            )   
        else:
            raise ValueError("Cannot infer ASR backend; set --asr-backend for a custom model path")
            
        segments_dataset = AudioDatasetForASR(annotation_dir=annotation_dir, target_sr=target_sr, speech_only=True)
        
        rank_dataset = torch.utils.data.Subset(
            segments_dataset,
            range(self.local_rank, len(segments_dataset), world_size),
        )
        
        self.segments_loader = torch.utils.data.DataLoader(
            rank_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=DataCollatorForASR(),
            pin_memory=True,
        )
        
    def generate(self):
        language_cache = {}
        # Preserve one empty output for silent recordings, including an all-silent run.
        if self.local_rank == 0:
            for annotation in sorted(self.annotation_dir.glob("*.jsonl")):
                if ".rank" not in annotation.name:
                    (self.output_dir / f"{annotation.stem}.rank0.jsonl").write_bytes(b"[]")
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
            f.write(json.dumps(seeks, ensure_ascii=True).encode("utf-8"))
            f.close()
            f = None

        from tqdm import tqdm
        for batch in tqdm(self.segments_loader, desc="Transcribing", unit="batch"):
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
            if isinstance(self.asr_model, WhisperASR):
                language_hints = []
                for path in audio_paths:
                    if path not in language_cache:
                        if self.asr_model.language:
                            language_cache[path] = self.asr_model.language
                        else:
                            context, _, _ = read_audio(
                                {"audio": path, "start": 0, "end": 30}, self.target_sr
                            )
                            if self.target_sr != 16000:
                                import librosa
                                context = librosa.resample(context.numpy(), orig_sr=self.target_sr, target_sr=16000)
                            else:
                                context = context.numpy()
                            language_cache[path] = self.asr_model.detect_language(context)
                            print(f"Detected language for {path}: {language_cache[path]}", flush=True)
                    language_hints.append(language_cache[path])
                texts, languages = self.asr_model.transcribe(audio=inputs, languages=language_hints)
            else:
                texts, languages = self.asr_model.transcribe(audio=inputs)
            if len(texts) != len(inputs) or len(languages) != len(inputs):
                raise RuntimeError("ASR returned a different number of results than input segments")
            for path, start, end, text, lang in zip(
                audio_paths, starts, ends, texts, languages
            ):
                key = f"{extract_bv_id_from_dir(Path(path).parent.stem)}_{Path(path).stem}.rank{self.local_rank}"

                if key != current_key:
                    close_file()
                    current_key = key
                    lengths = []

                    out_file = self.output_dir / f"{key}.jsonl"
                    f = open(out_file, "wb")

                line = json.dumps(
                    {
                        "audio_path": path,
                        "start": start.item(),
                        "end": end.item(),
                        "text": text.strip(),
                        "language": (lang or "unknown").strip(),
                    },
                    ensure_ascii=False,
                ) + "\n"

                lengths.append(len(line.encode("utf-8")))
                f.write(line.encode("utf-8"))

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
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        rank=rank,
        language=args.language,
        asr_backend=args.asr_backend,
        num_workers=args.num_workers,
        device=args.device,
        world_size=world_size
    )
    annotator.generate()
    
if __name__ == "__main__":
    run()
