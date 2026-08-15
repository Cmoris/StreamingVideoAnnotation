"""Self-contained tests for every dataset/collator pair in ``bin.data``.

Run from the repository root with::

    python bin/data/test_dataset_collators.py

The tests use temporary metadata and deterministic decoder stubs, so no real
audio files, model downloads, or GPU are required.
"""

from __future__ import annotations

import json
import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader


# Executing this file directly adds bin/ to sys.path, making data importable as
# a package and preserving dataset.py's relative imports.
BIN_DIR = Path(__file__).resolve().parents[1]
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))

from data import dataset as dataset_module  # noqa: E402
from data.collator import (  # noqa: E402
    DataCollatorForASR,
    DataCollatorForAudio,
    DataCollatorForCaption,
    DataCollatorForTranscripts,
)
from data.dataset import (  # noqa: E402
    AudioDatasetForASR,
    AudioDatasetForCaption,
    AudioDatasetForSegment,
    TranscriptDataset,
)


def _write_indexed_jsonl(path: Path, records: list[dict]) -> None:
    """Write the seek-index format consumed by the annotation datasets."""
    seeks: list[int] = []
    with path.open("wb") as file:
        for record in records:
            seeks.append(file.tell())
            file.write((json.dumps(record) + "\n").encode("utf-8"))
        file.write((json.dumps(seeks) + "\n").encode("utf-8"))


def _fake_read_audio(ele: dict, target_sr: int = 16_000, bounds_in_samples: bool = False):
    start = float(ele.get("start") or 0)
    end = ele.get("end")
    if end is None:
        length = 8
    elif bounds_in_samples:
        length = max(1, int(float(end) - start))
    else:
        length = max(1, round((float(end) - start) * target_sr))
    waveform = torch.arange(length, dtype=torch.float32)
    pts = start + np.arange(length) / target_sr
    return waveform, pts, target_sr


class _FakeProcessor:
    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        assert conversation and add_generation_prompt and not tokenize
        return "audio prompt"

    def __call__(self, text, audio, return_tensors, use_audio_in_video):
        assert text and return_tensors == "pt" and not use_audio_in_video
        time_steps = max(1, len(audio) // 2)
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "input_features": torch.arange(
                2 * time_steps, dtype=torch.float32
            ).reshape(1, 2, time_steps),
            "feature_attention_mask": torch.ones((1, time_steps), dtype=torch.long),
        }


class DatasetCollatorTests(unittest.TestCase):
    def test_segment_dataset_with_audio_collator(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            (audio_dir / "short.wav").touch()
            (audio_dir / "long.mp3").touch()

            with patch.object(dataset_module, "read_audio", side_effect=lambda ele, target_sr: (
                torch.arange(3 if Path(ele["audio"]).stem == "short" else 5, dtype=torch.float32),
                np.array([]),
                target_sr,
            )):
                dataset = AudioDatasetForSegment(audio_dir, target_sr=16_000)
                batch = next(iter(DataLoader(
                    dataset, batch_size=2, collate_fn=DataCollatorForAudio()
                )))

        self.assertEqual(tuple(batch["input_values"].shape), (2, 5))
        self.assertEqual(sorted(batch["lengths"].tolist()), [3, 5])
        self.assertEqual(batch["attention_mask"].sum().item(), 8)
        self.assertEqual(batch["sampling_rate"].tolist(), [16_000, 16_000])

    def test_asr_dataset_with_asr_collator(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotation_dir = Path(tmp)
            _write_indexed_jsonl(annotation_dir / "segments.jsonl", [
                {"audio_path": "a.wav", "type": "speech", "start": 0, "end": 4},
                {"audio_path": "b.wav", "type": "environment_event", "start": 2, "end": 9},
            ])

            with patch.object(dataset_module, "read_audio", side_effect=_fake_read_audio):
                dataset = AudioDatasetForASR(annotation_dir)
                batch = next(iter(DataLoader(
                    dataset, batch_size=2, collate_fn=DataCollatorForASR()
                )))

        self.assertEqual(tuple(batch["waveforms"].shape), (2, 7))
        self.assertEqual(batch["attention_mask"].sum(dim=1).tolist(), [4, 7])
        self.assertEqual(batch["type"], ["speech", "environment_event"])
        self.assertEqual(batch["start"].tolist(), [0, 2])
        self.assertEqual(batch["end"].tolist(), [4, 9])

    def test_caption_dataset_with_caption_collator(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotation_dir = Path(tmp)
            _write_indexed_jsonl(annotation_dir / "captions.jsonl", [
                {"audio_path": "a.wav", "type": "environment_event", "start": 0, "end": 4},
                {"audio_path": "b.wav", "type": "background_ambience", "start": 0, "end": 8},
            ])

            with patch.object(dataset_module, "read_audio", side_effect=_fake_read_audio):
                dataset = AudioDatasetForCaption(
                    annotation_dir, processor=_FakeProcessor()
                )
                batch = next(iter(DataLoader(
                    dataset, batch_size=2, collate_fn=DataCollatorForCaption()
                )))

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(tuple(batch["input_features"].shape), (2, 2, 4))
        self.assertEqual(batch["feature_attention_mask"].sum(dim=1).tolist(), [2, 4])

    def test_transcript_dataset_with_transcripts_collator(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotation_dir = Path(tmp)
            _write_indexed_jsonl(annotation_dir / "transcripts.jsonl", [
                {"audio_path": "a.wav", "text": "hello", "start": 0.0,
                 "end": 0.25, "language": "English"},
                {"audio_path": "a.wav", "text": "你好", "start": 0.25,
                 "end": 0.5, "language": "Chinese"},
            ])

            with patch.object(dataset_module, "read_audio", side_effect=_fake_read_audio):
                dataset = TranscriptDataset(annotation_dir)
                batch = next(iter(DataLoader(
                    dataset, batch_size=2, collate_fn=DataCollatorForTranscripts()
                )))

        self.assertEqual(tuple(batch["waveforms"].shape), (2, 8))
        self.assertEqual(batch["waveform_lengths"].tolist(), [8, 8])
        self.assertEqual(batch["language"], ["en", "zh"])
        self.assertEqual([item["text"] for item in batch["segments"]], ["hello", "你好"])


def _add_common_loader_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0)")
    parser.add_argument(
        "--max-batches", type=int, default=1,
        help="Maximum batches to read; use 0 for all batches (default: 1)",
    )
    parser.add_argument("--target-sr", type=int, default=16_000, help="Target sample rate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test dataset classes and their corresponding collators with your own data."
    )
    subparsers = parser.add_subparsers(dest="dataset_type", required=True)

    self_test = subparsers.add_parser("self-test", help="Run tests using generated temporary data")
    self_test.add_argument("--verbosity", type=int, default=2, choices=(0, 1, 2))

    segment = subparsers.add_parser("segment", help="AudioDatasetForSegment + DataCollatorForAudio")
    segment.add_argument("--audio-dir", type=Path, required=True)
    _add_common_loader_args(segment)

    for name, help_text in (
        ("asr", "AudioDatasetForASR + DataCollatorForASR"),
        ("transcript", "TranscriptDataset + DataCollatorForTranscripts"),
    ):
        child = subparsers.add_parser(name, help=help_text)
        child.add_argument("--annotation-dir", type=Path, required=True)
        _add_common_loader_args(child)

    caption = subparsers.add_parser(
        "caption", help="AudioDatasetForCaption + DataCollatorForCaption"
    )
    caption.add_argument("--annotation-dir", type=Path, required=True)
    caption.add_argument(
        "--processor-path", required=True,
        help="Local processor directory or Hugging Face model ID",
    )
    caption.add_argument(
        "--trust-remote-code", action="store_true",
        help="Allow custom processor code when loading --processor-path",
    )
    _add_common_loader_args(caption)
    return parser


def _describe(value, indent: str = "  ") -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, list):
        preview = repr(value[:2])
        return f"list(len={len(value)}, preview={preview})"
    if isinstance(value, tuple):
        return f"tuple(len={len(value)})"
    if isinstance(value, dict):
        return "dict(keys=" + repr(list(value)) + ")"
    return repr(value)


def _print_batch(batch_index: int, batch) -> None:
    print(f"\nBatch {batch_index}:")
    if isinstance(batch, dict):
        for key, value in batch.items():
            print(f"  {key}: {_describe(value)}")
    else:
        print(f"  {_describe(batch)}")


def run_with_user_data(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_batches < 0:
        raise ValueError("--max-batches cannot be negative")

    if args.dataset_type == "segment":
        dataset = AudioDatasetForSegment(args.audio_dir, target_sr=args.target_sr)
        collator = DataCollatorForAudio()
    elif args.dataset_type == "asr":
        dataset = AudioDatasetForASR(args.annotation_dir, target_sr=args.target_sr)
        collator = DataCollatorForASR()
    elif args.dataset_type == "transcript":
        dataset = TranscriptDataset(args.annotation_dir, target_sr=args.target_sr)
        collator = DataCollatorForTranscripts()
    elif args.dataset_type == "caption":
        from transformers import AutoProcessor

        print(f"Loading processor: {args.processor_path}")
        processor = AutoProcessor.from_pretrained(
            args.processor_path, trust_remote_code=args.trust_remote_code
        )
        dataset = AudioDatasetForCaption(
            args.annotation_dir, target_sr=args.target_sr, processor=processor
        )
        pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", 0)
        collator = DataCollatorForCaption(pad_token_id=pad_token_id or 0)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    print(f"Dataset: {type(dataset).__name__}")
    print(f"Collator: {type(collator).__name__}")
    print(f"Samples: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("The dataset is empty; check the supplied directory and annotation files")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        shuffle=False,
    )
    batches_read = 0
    for batch_index, batch in enumerate(loader, start=1):
        _print_batch(batch_index, batch)
        batches_read += 1
        if args.max_batches and batches_read >= args.max_batches:
            break
    print(f"\nPASS: successfully loaded and collated {batches_read} batch(es).")


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset_type == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DatasetCollatorTests)
        result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
        return 0 if result.wasSuccessful() else 1

    try:
        run_with_user_data(args)
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
