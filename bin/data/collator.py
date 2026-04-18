import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Dict, Tuple, Any

class DataCollatorForVideo:
    def __init__(self, padding_value=0.0):
        self.padding_value = padding_value
    
    def __call__(self, batch):
        """
        batch: List[(waveform, sr)]
        """
        waveforms = [torch.from_numpy(w).float() for _, w, _ in batch]
        srs = [torch.tensor(sr, dtype=torch.long) for _, _, sr in batch]
        audio_paths = [a for a, _, _ in batch]
        lengths = torch.tensor([w.size(0) for w in waveforms], dtype=torch.long)
        batch_size = len(waveforms)

        max_len = max(lengths).item()

        padded = torch.full(
            (batch_size, max_len),
            self.padding_value,
            dtype=torch.float32,
        )
        
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
        )
        
        srs = torch.stack(srs, dim=0)

        for i, w in enumerate(waveforms):
            padded[i, : w.size(0)] = w
            attention_mask[i, : w.size(0)] = 1

        return {
            "input_values": padded,   # [B, T]
            "attention_mask": attention_mask, # [B, T]
            "sampling_rate": srs,    # [B] 
            "lengths": lengths,       # [B]
            "audio_path": audio_paths # [B]
        }

class DataCollatorForAudio:
    def __init__(self, padding_value=0.0):
        self.padding_value = padding_value

    def __call__(self, batch):
        """
        batch: List[(waveform, sr)]
        """
        waveforms = [torch.from_numpy(w).float() for _, w, _ in batch]
        srs = [torch.tensor(sr, dtype=torch.long) for _, _, sr in batch]
        audio_paths = [a for a, _, _ in batch]
        lengths = torch.tensor([w.size(0) for w in waveforms], dtype=torch.long)
        batch_size = len(waveforms)

        max_len = max(lengths).item()

        padded = torch.full(
            (batch_size, max_len),
            self.padding_value,
            dtype=torch.float32,
        )
        
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
        )
        
        srs = torch.stack(srs, dim=0)

        for i, w in enumerate(waveforms):
            padded[i, : w.size(0)] = w
            attention_mask[i, : w.size(0)] = 1

        return {
            "input_values": padded,   # [B, T]
            "attention_mask": attention_mask, # [B, T]
            "sampling_rate": srs,    # [B] 
            "lengths": lengths,       # [B]
            "audio_path": audio_paths # [B]
        }

class DataCollatorForASR:
    def __init__(
        self,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
    ):
        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        features: List of dataset outputs
        """
        waveforms = [f["waveform"] for f in features]
        srs = [torch.tensor(f["sr"], dtype=torch.long) for f in features]
        starts = [torch.tensor(f["start"], dtype=torch.float32) for f in features]
        ends = [torch.tensor(f["end"], dtype=torch.float32) for f in features]

        lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
        max_len = lengths.max().item()

        batch_size = len(waveforms)

        input_values = torch.full(
            (batch_size, max_len),
            fill_value=self.padding_value,
            dtype=torch.float32,
        )

        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
        )
        
        srs = torch.stack(srs, dim=0)
        starts = torch.stack(starts, dim=0)
        ends = torch.stack(ends, dim=0)

        for i, w in enumerate(waveforms):
            input_values[i, : len(w)] = w
            attention_mask[i, : len(w)] = 1

        batch = {
            "waveforms": input_values,
            "type": [f["audio_type"] for f in features],
            "sampling_rate": srs,
            "start": starts,
            "end": ends,
            "audio_path": [f["audio_path"] for f in features],
        }

        if self.return_attention_mask:
            batch["attention_mask"] = attention_mask

        return batch

class DataCollatorForCausalLM:
    def __init__(
        self,
        pad_token_id: int,
    ):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        lengths = torch.tensor([len(ids) for ids in input_ids], dtype=torch.long)
        max_len = lengths.max().item()
        batch_size = len(input_ids)

        batch_input_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        batch_attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
        )

        for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
            seq_len = len(ids)
            batch_input_ids[i, :seq_len] = ids
            batch_attention_mask[i, :seq_len] = mask

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
        }



class DataCollatorForCaption:
    def __init__(
        self,
        pad_token_id: int = 0,
        feature_padding_value: float = 0.0,
        label_pad_token_id: int = -100,
    ):
        self.pad_token_id = pad_token_id
        self.feature_padding_value = feature_padding_value
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        # ===== text =====
        input_ids = [f["input_ids"].squeeze(0) for f in features]
        attention_mask = [f["attention_mask"].squeeze(0) for f in features]

        batch_input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        ).to(torch.long)
        
        batch_attention_mask = pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        ).to(torch.long)

        input_features = [f["input_features"].squeeze(0) for f in features]
        feature_attention_mask = [f["feature_attention_mask"].squeeze(0) for f in features]
        
        batch_size = len(input_features)
        lengths = torch.tensor([features.size(1) for features in input_features], dtype=torch.long)
        dim = input_features[0].size(0)
        max_len = lengths.max().item()
        
        # pad along time dimension (T)
        batch_input_features = torch.full(
            (batch_size, dim, max_len),
            fill_value=self.feature_padding_value,
            dtype=torch.float,
        )
        
        for i, features in enumerate(input_features):
            seq_len = features.size(1)
            batch_input_features[i, :, :seq_len] = features
        
        batch_feature_attention_mask = pad_sequence(
            feature_attention_mask,
            batch_first=True,
            padding_value=0,
        ).to(torch.long)

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "input_features": batch_input_features,
            "feature_attention_mask": batch_feature_attention_mask,
        }

        return batch
        
class DataCollatorForTranscripts:

    pad_value: float = 0.0

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, str, List[Dict[str, Any]]]],
    ) -> Dict[str, Any]:

        waveforms, audio_paths, segments, language = zip(*batch)
        # Record original lengths (number of samples, channel-agnostic)
        waveform_lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)

        padded = pad_sequence(
            waveforms,
            batch_first=True,
            padding_value=self.pad_value,
        )  # [B, T_max]

        return {
            "waveforms":        padded,
            "waveform_lengths": waveform_lengths,
            "audio_paths":      list(audio_paths),
            "segments":         list(segments),
            "language":         list(language)
        }