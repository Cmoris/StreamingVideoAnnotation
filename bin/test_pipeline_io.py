"""Regression tests for UTF-8 byte offsets and silent audio, without ML dependencies."""
import json
import tempfile
import unittest
from pathlib import Path

from merge import merge_rank_jsonls, read_data


class PipelineIOTests(unittest.TestCase):
    def test_merge_preserves_unicode_and_byte_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            for rank, start in enumerate((2, 1)):
                path = root / f"audio.rank{rank}.jsonl"
                record = {"audio_path": "中文.wav", "start": start,
                          "end": start + 0.25, "text": "你好、日本語"}
                path.write_bytes((json.dumps(record, ensure_ascii=False) + "\n[0]").encode("utf-8"))
                files.append(path)
            output = root / "merged.jsonl"
            merge_rank_jsonls(files, output)
            self.assertEqual([r["start"] for r in read_data(output)], [1, 2])
            offsets = json.loads(output.read_bytes().splitlines()[-1])
            with output.open("rb") as stream:
                for offset, start in zip(offsets, (1, 2)):
                    stream.seek(offset)
                    record = json.loads(stream.readline())
                    self.assertEqual(record["start"], start)
                    self.assertEqual(record["text"], "你好、日本語")

    def test_empty_annotations_merge_to_empty_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "silent.rank0.jsonl"
            source.write_bytes(b"[]")
            output = root / "silent.jsonl"
            merge_rank_jsonls([source], output)
            self.assertEqual(output.read_bytes(), b"[]")
            self.assertEqual(list(read_data(output)), [])


if __name__ == "__main__":
    unittest.main()
