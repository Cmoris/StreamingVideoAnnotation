import torch
import librosa
import numpy as np
from typing import List, Tuple
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor
from qwen_asr import Qwen3ASRModel


class WhisperASR:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        language: str = "ja",
        device: str = "cuda",
    ):
        self.language = language

        self.device = device

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map=device,
                # attn_implementation="flash_attention_2",
            )

        self.model.eval()

        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe"
        )

    @torch.no_grad()
    def transcribe(self, waveforms: List[Tuple[np.ndarray, int]]) -> List[str]:
        waves, srs = [], []
        for waveform, sr in waveforms:
            waves.append(waveform)
            srs.append(sr)
        
        waves = torch.stack(waves, dim=0)
        
        inputs = self.processor(
            waves,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        
        input_features = inputs.input_features.to(device=self.device, dtype=torch.bfloat16)

        predicted_ids = self.model.generate(
            input_features,
            forced_decoder_ids=self.forced_decoder_ids,
        )

        texts = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        return texts
    
    def detect_language(self, audio: np.ndarray) -> str:
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs.input_features,
                max_new_tokens=1  # 只生成语言 token
            )

        # 解码第一个生成的 token 即为语言
        lang_token = self.processor.tokenizer.decode(predicted_ids[0][1])  # [0]=<|startoftranscript|>
        return lang_token
    
    
class QwenASR:
    def __init__(self, 
                 model_name="Qwen/Qwen3-ASR-1.7B", 
                 language=None, 
                 device="cuda:0"):
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

