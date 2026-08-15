import torch
import logging
import numpy as np
from typing import List, Union
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

logger = logging.getLogger(__name__)

class Qwen3OmniCaptioner:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        logger.info(f"Loading Qwen3-Omni model from {model_name}...")
        
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device,
            trust_remote_code=True
        ).eval()
        
    def _get_prompt_text(self, segment_type: str) -> str:
        if segment_type == "environment_event":
            return "Identify the sound event and action. Short answer."
        elif segment_type == "background_ambience":
            return "Briefly describe the background scene and atmosphere. In one sentence."
        else:
            return "Describe this audio concisely."

    def generate(self, waveforms: Union[torch.Tensor, List[np.ndarray]], types: List[str]) -> List[str]:
        """
        [串行兼容版] 逐条处理音频，规避模型不支持 Batch Audio 的问题。
        
        Args:
            waveforms: (N, T) Tensor 或 List[np.ndarray]
            types: List[str]
            
        Returns:
            List[str]: 顺序对应的 Caption 列表
        """
        results = []
        
        # 0. 统一转为 List[np.ndarray]，方便逐条取用
        if isinstance(waveforms, torch.Tensor):
            # 将 Tensor 拆分为 list，并转 numpy
            audio_inputs = [w.cpu().float().numpy() for w in waveforms]
        else:
            audio_inputs = waveforms

        # 1. 串行循环处理
        for i, (audio, t) in enumerate(zip(audio_inputs, types)):
            try:
                # --- A. 构建单条 Conversation ---
                prompt_content = self._get_prompt_text(t)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": "placeholder"},
                            {"type": "text", "text": prompt_content},
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
                
                # --- B. 单条处理 (注意 audio 参数传入的是含有一个元素的 list) ---
                # 即使是单条，processor 通常也期望 audio 参数是 list 格式: [audio_data]
                inputs = self.processor(
                    text=[text],           # list of 1
                    audio=[audio],         # list of 1
                    return_tensors="pt",
                    use_audio_in_video=False
                )
                
                inputs = inputs.to(self.model.device).to(self.model.dtype)
                
                # --- C. 单条推理 ---
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=48,
                        min_new_tokens=2,
                        do_sample=False,
                        thinker_return_dict_in_generate=True 
                    )

                # --- D. 提取与解码 ---
                output_ids = self._extract_output_ids(outputs)
                input_len = inputs["input_ids"].shape[1]
                generated_ids = output_ids[:, input_len:]
                
                caption = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0] # 取第一个结果
                
                results.append(caption.strip())

            except Exception as e:
                logger.error(f"Error processing item {i} in batch: {e}")
                import traceback
                traceback.print_exc()
                # 遇到单条报错，填入空值或错误提示，保证列表长度对齐
                results.append("Error generating caption")

        return results

    def _extract_output_ids(self, outputs) -> torch.Tensor:
        """健壮的返回值提取逻辑"""
        output_ids = None
        if hasattr(outputs, 'sequences'):
            output_ids = outputs.sequences
        elif isinstance(outputs, (tuple, list)):
            first_element = outputs[0]
            if isinstance(first_element, torch.Tensor):
                output_ids = first_element
            elif hasattr(first_element, 'sequences'):
                output_ids = first_element.sequences
            else:
                output_ids = first_element
        elif isinstance(outputs, torch.Tensor):
            output_ids = outputs

        if output_ids is None:
             if hasattr(outputs, 'to_tuple'):
                 output_ids = outputs.to_tuple()[0]
        
        return output_ids
