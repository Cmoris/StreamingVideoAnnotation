import os
import sys
import torch
import numpy as np
import torch.nn as nn
from omegaconf import OmegaConf

from maai.models.vap import VapGPT
from maai.models.config import VapConfig as VapModelConfig

import librosa

class VAP:

    def __init__(self, config):

        self.config = config
        self.device = self.config.device
        
        # environment 
        self._setup_environment()
        
        # 内部状态：用于跟踪已处理的音频位置
        self.processed_frame_count = 0
        self.context_frames = []  
        
        # KV Cache
        self.vap_cache = None
        
        # Load VAP model
        # print(f"[VAP] Load VAP model")
        self._load_model()
    
    def _setup_environment(self):

        env_vars = {
            'HF_HUB_OFFLINE': self.config.environment.HF_HUB_OFFLINE,
            'TRANSFORMERS_OFFLINE': '1',
            'HF_HOME': self.config.environment.HF_HOME,
            'HF_HUB_CACHE': self.config.environment.HF_HUB_CACHE
        }
        
        os.environ.update(env_vars)
    
    def _load_cpc_encoder(self):
        
        cpc_path = self.config.model.cpc_model_path
        self.vap.load_encoder(cpc_model=cpc_path)

    
    def _load_downsample_params(self, state_dict):
        if 'encoder.downsample.1.weight' not in state_dict:
            return
        
        for encoder in [self.vap.encoder1, self.vap.encoder2]:
            encoder.downsample[1].weight = nn.Parameter(state_dict['encoder.downsample.1.weight'])
            encoder.downsample[1].bias = nn.Parameter(state_dict['encoder.downsample.1.bias'])
            encoder.downsample[2].ln.weight = nn.Parameter(state_dict['encoder.downsample.2.ln.weight'])
            encoder.downsample[2].ln.bias = nn.Parameter(state_dict['encoder.downsample.2.ln.bias'])
    
    def _load_model(self):

        if self.config.model.vap_source_path not in sys.path:
            sys.path.insert(0, self.config.model.vap_source_path)
        
        vap_config = VapModelConfig()
        self.vap = VapGPT(vap_config)

        state_dict = torch.load(self.config.model.local_model_path, map_location=self.device)
        
        self._load_cpc_encoder()
        self.vap.load_state_dict(state_dict, strict=False)
        self._load_downsample_params(state_dict)
        
        self.vap.to(self.device)
        self.vap.eval()
    
    def process_audio(self, audio_frame):
        
        x1_proc = audio_frame
        x2_proc = np.zeros_like(x1_proc)
        result = self._inference(x1_proc, x2_proc)

        return result
    
    def _inference(self, x1_proc, x2_proc):
        with torch.no_grad():
            
            x1_ = torch.from_numpy(x1_proc).float().unsqueeze(0).unsqueeze(0)
            x2_ = torch.from_numpy(x2_proc).float().unsqueeze(0).unsqueeze(0)
            
            if self.device != 'cpu':
                x1_ = x1_.to(self.device, non_blocking=True)
                x2_ = x2_.to(self.device, non_blocking=True)
            
            e1, e2 = self.vap.encode_audio(x1_, x2_)
            
            if self.config.cache.use_kv_cache:
                out, self.vap_cache = self.vap.forward(e1, e2, cache=self.vap_cache)
                if self.vap_cache is not None:
                    self._trim_cache()
            else:
                out, _ = self.vap.forward(e1, e2, cache=None)
            
            return out.get('p_now')[-1]

    def _trim_cache(self):

        if self.vap_cache is None:
            return
        
        max_len = self.config.vap.context_len_sec - 1
        new_cache = {}
        
        for key, (k_list, v_list) in self.vap_cache.items():
            new_k_list = [
                t[..., -max_len:, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t 
                for t in k_list
            ]
            new_v_list = [
                t[..., -max_len:, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t 
                for t in v_list
            ]
            new_cache[key] = (new_k_list, new_v_list)
        
        self.vap_cache = new_cache
        
        
if __name__ == "__main__":
    config_path = '/home/muyun/project/Annotation/bin/config.yaml'
    config = OmegaConf.load(config_path)
    vap_model = VAP(config)
    audio_path = "/home/muyun/project/Annotation/sources/audio_mp4a.40.5_44809.mp4"
    waveform, sr = librosa.load(audio_path)
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)