import torch
import librosa
import numpy as np
from typing import List, Tuple
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor



class WhisperASR:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        language: str = None,
        device: str = "cuda",
    ):
        self.language = language

        self.device = device
        self.dtype = torch.float32 if str(device).startswith("cpu") else torch.bfloat16

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                dtype=self.dtype,
                # attn_implementation="flash_attention_2",
            ).to(device)

        self.model.eval()

        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.language = language
        self.model.generation_config.task = "transcribe"

    @torch.no_grad()
    def transcribe(self, audio: List[Tuple[np.ndarray, int]], languages=None) -> List[str]:
        waves, srs = [], []
        for waveform, sr in audio:
            if sr != 16000:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            waves.append(waveform)
            srs.append(sr)
        
        inputs = self.processor(
            waves,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=False,
            return_attention_mask=True,
        )
        
        input_features = inputs.input_features.to(device=self.device, dtype=self.model.dtype)
        if languages is not None:
            if len(languages) != len(waves):
                raise ValueError("Expected one language per input waveform")
        elif self.language:
            languages = [self.language] * len(waves)
        else:
            language_ids = self.model.detect_language(input_features)
            languages = [token[2:-2] for token in
                         self.processor.tokenizer.convert_ids_to_tokens(language_ids.tolist())]

        predicted_ids = self.model.generate(
            input_features,
            attention_mask=inputs.attention_mask.to(self.device),
            return_timestamps=True,
            language=languages,
        )

        texts = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        return texts, languages
    
    def detect_language(self, audio: np.ndarray) -> str:
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        with torch.no_grad():
            language_ids = self.model.detect_language(
                inputs.input_features.to(device=self.device, dtype=self.model.dtype)
            )
        return self.processor.tokenizer.convert_ids_to_tokens(language_ids.tolist())[0][2:-2]
    
    
class QwenASR:
    def __init__(self, 
                 model_name="Qwen/Qwen3-ASR-1.7B", 
                 language=None,
                 device="cuda:0"):
        from qwen_asr import Qwen3ASRModel
        self.model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
            # attn_implementation="flash_attention_2",
            max_inference_batch_size=32,
            max_new_tokens=512,
        )

        self.language = language
        
    def transcribe(self, audio: List[Tuple[np.ndarray, int]]) -> List[str]:
        results = self.model.transcribe(
            audio=audio,
            language=self.language,
        )
        texts = [result.text for result in results]
        languages = [result.language for result in results]
        
        return texts, languages
    

if __name__ == "__main__":
    from vad_model import VAD
    asr = WhisperASR(
        model_name="/mnt/nvme/workspaces/muyun/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1",
        language=None, 
    )
    audio_path = "/home/muyun/project/Annotation/sources/audio_mp4a.40.5_44809.mp4"
    waveform, sr = librosa.load(audio_path)
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    vad = VAD()
    speech_segments = vad.generate(waveform, 16000)
    for start_sample, end_sample in speech_segments:
        print(f"Speech segment: {start_sample}-{end_sample}")
        segment_waveform = waveform[start_sample:end_sample]
        results = asr.generate([[0, start_sample/16000, end_sample/16000, segment_waveform]])
        print(results)
