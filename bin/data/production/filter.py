import json
from pathlib import Path

dir = "/mnt/nvme/workspaces/muyun/Dataset/livecc/all_stream"
jsonl_files = [x for x in Path(dir).glob("*.jsonl")]
all_lines = []
for jsonl_file in jsonl_files:
    with open(jsonl_file) as f:
        lines = f.readlines()[:-1]
        all_lines.extend([json.loads(line) for line in lines])


