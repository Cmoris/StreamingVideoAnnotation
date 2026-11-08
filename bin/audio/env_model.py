import torch
import torchaudio
from typing import List, Tuple

class EnvDetect:
    def __init__(self, 
                 sampling_rate,
                 device
                ):
        
        self.sampling_rate = sampling_rate
        self.device = device
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            normalized=True
        ).to(device)
    
    def _calculate_overlap(self, seg1, seg2):
        start = max(seg1[0], seg2[0])
        end = min(seg1[1], seg2[1])
        return max(0, end - start)

    def detect_environmental_events(self, 
                                    waveforms: torch.Tensor, 
                                    speech_segments: List[Tuple], 
                                    onset_threshold_alpha: float=1.5, 
                                    onset_backoff_sec: float=1.0):
        """
        全 GPU 实现的环境音突发检测
        waveform_tensor: (1, T) or (T,) on GPU
        speech_segments: List of (start, end)
        """
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0) # (1, T)
        waveforms = waveforms.to(self.device)
        if waveforms.shape[-1] <= self.mel_transform.n_fft // 2:
            return []
        
        # 1. 计算 Log-Mel 频谱 (模拟人耳感知)
        # (1, n_mels, n_frames)
        mel_spec = self.mel_transform(waveforms) 
        # Log 压缩 (加微小值防止 log(0))
        log_mel = torch.log(mel_spec + 1e-6)

        # 2. 计算 Spectral Flux (频谱通量/能量变化率)
        # 计算相邻帧的差值 (Diff)
        diff = log_mel[:, :, 1:] - log_mel[:, :, :-1]
        # 仅保留正向变化 (Relu)，即能量增强的时刻（Onset）
        flux = torch.relu(diff)
        # 在频率维度求平均，得到随时间变化的 Onset Envelope (1, n_frames)
        onset_env = flux.mean(dim=1)

        # 3. 峰值检测 (Peak Picking)
        # 使用 MaxPool1d 寻找局部最大值 (窗口大小设为 5 帧，约 50ms)
        kernel_size = 5
        padding = kernel_size // 2
        # padding 保证输出长度不变
        max_filtered = torch.nn.functional.max_pool1d(
            onset_env.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=padding
        ).squeeze(0)
        
        # 动态阈值: 平均值 + alpha * 标准差
        threshold = onset_env.mean() + onset_threshold_alpha * onset_env.std()
        
        # 寻找既是局部最大值又超过阈值的点
        is_peak = (onset_env == max_filtered) & (onset_env > threshold)
        
        # 获取帧索引
        onset_frames = torch.nonzero(is_peak, as_tuple=True)[1]
        
        # 转换为样本点 (Sample Index)
        # hop_length = 512
        onset_samples = (onset_frames + 1) * 512
        
        # 4. 矩阵化过滤：剔除落在人声区间的点
        valid_events = []
        backoff_samples = int(onset_backoff_sec * self.sampling_rate)
        
        if len(speech_segments) > 0 and len(onset_samples) > 0:
            # 构建人声区间 Tensor: (N_speech, 2)
            speech_intervals = torch.tensor(speech_segments, device=self.device)
            
            # 广播比较: (N_onsets, B) vs (B, N_speech)
            # 检查每个 onset 是否 >= 任意 speech_start 且 <= 对应 speech_end
            starts = speech_intervals[:, 0] # (B, N_speech)
            ends = speech_intervals[:, 1]   # (B, N_speech)
            points = onset_samples.unsqueeze(1)          # (N_onsets, 1)
            
            # inside_mask: (N_onsets, N_speech)
            inside_mask = (points >= starts) & (points <= ends)
            
            # 如果某个 onset 落在任意一个人声段内 (any=True)，则视为 True
            is_inside_speech = inside_mask.any(dim=1)
            
            # 取反，保留不在人声内的点
            valid_onsets = onset_samples[~is_inside_speech]
        else:
            valid_onsets = onset_samples

        # 5. 转换为区间 (Start, End) 并转回 List
        for t in valid_onsets:
            t = t.item()
            start = max(0, t - backoff_samples)
            end = min(waveforms.shape[1], t + backoff_samples)
            valid_events.append((start, end))
            
        return self._merge_segments(valid_events)

    def _merge_segments(self, segments):
        if not segments: return []
        segments.sort(key=lambda x: x[0])
        merged = [segments[0]]
        for curr in segments[1:]:
            prev = merged[-1]
            if curr[0] <= prev[1]:
                merged[-1] = (prev[0], max(prev[1], curr[1]))
            else:
                merged.append(curr)
        return merged

    def generate_background_windows(self, total_samples, bg_window_sec=8.0):
        window_size = int(bg_window_sec * self.sampling_rate)
        step_size = window_size // 2 # 50% overlap
        windows = []
        for start in range(0, total_samples - window_size + 1, step_size):
            windows.append((start, start + window_size))
        return windows
    
    def generate(self, waveforms, speech_segments):
        env_segments = self.detect_environmental_events(waveforms, speech_segments)
            
        raw_bg_segments = self.generate_background_windows(len(waveforms))
        
        occupied_segments = self._merge_segments(speech_segments + env_segments)
        filtered_bg_segments = []

        for bg_s, bg_e in raw_bg_segments:
            bg_len = bg_e - bg_s
            total_overlap = 0
            for occ_s, occ_e in occupied_segments:
                total_overlap += self._calculate_overlap((bg_s, bg_e), (occ_s, occ_e))
            
            if (total_overlap / bg_len) < 0.5:
                filtered_bg_segments.append((bg_s, bg_e))
                
        return env_segments, filtered_bg_segments

