import json
from pathlib import Path
import decord
from transformers import logging

logger = logging.get_logger(__name__)

def safe_video_reader(video_path):
    try:
        vr = decord.VideoReader(video_path, num_threads=2)
        return True
    except Exception as e:
        logger.warning(f"Failed try to get item {video_path}: {e}")
        return False
    
jsonl_path = "/home/muyun/project/Annotation/all_results/livecc/livecc_english_10.jsonl"
data = [json.loads(line) for line in open(jsonl_path).readlines()[:-1]]
video_paths = set([conv[0]["content"][0]["video"] for conv in data])
error_video_paths = []
for video_path in video_paths:
    if not safe_video_reader(video_path):
        error_video_paths.append(video_path)
        
print(error_video_paths)