"""Whisper regression test without downloading weights or using a GPU."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from audio.asr_model import WhisperASR


class WhisperTests(unittest.TestCase):
    def test_language_detection_does_not_depend_on_generated_prefix(self):
        backend = WhisperASR.__new__(WhisperASR)
        backend.language = None
        backend.device = "cpu"
        backend.processor = MagicMock()
        backend.processor.return_value = SimpleNamespace(
            input_features=torch.zeros(1, 128, 3100),
            attention_mask=torch.ones(1, 3100, dtype=torch.long),
        )
        backend.processor.tokenizer.convert_ids_to_tokens.return_value = ["<|ja|>"]
        backend.processor.batch_decode.return_value = ["こんにちは"]
        backend.model = MagicMock()
        backend.model.dtype = torch.float32
        backend.model.detect_language.return_value = torch.tensor([50266])
        # Timestamp generation may strip the language and decoder prompt tokens.
        backend.model.generate.return_value = torch.tensor([[50365, 123, 50370]])
        texts, languages = backend.transcribe([(np.zeros(31 * 16000), 16000)])
        self.assertEqual(texts, ["こんにちは"])
        self.assertEqual(languages, ["ja"])
        self.assertEqual(backend.model.generate.call_args.kwargs["language"], ["ja"])
        backend.model.detect_language.assert_called_once()
        self.assertFalse(backend.processor.call_args.kwargs["truncation"])
        self.assertTrue(backend.model.generate.call_args.kwargs["return_timestamps"])
        self.assertEqual(tuple(backend.model.generate.call_args.kwargs["attention_mask"].shape), (1, 3100))
        backend.model.detect_language.reset_mock()
        texts, languages = backend.transcribe([(np.zeros(16000), 16000)], languages=["ja"])
        backend.model.detect_language.assert_not_called()
        self.assertEqual(languages, ["ja"])
        self.assertEqual(backend.detect_language(np.zeros(16000)), "ja")
        self.assertEqual(backend.processor.call_args.kwargs["padding"], "max_length")
        self.assertTrue(backend.processor.call_args.kwargs["truncation"])


if __name__ == "__main__":
    unittest.main()
