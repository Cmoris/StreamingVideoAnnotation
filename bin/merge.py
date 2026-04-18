import json
import argparse
from pathlib import Path
from typing import List
from collections import defaultdict

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None
    )


def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2) # avoid last \n
        while f.read(1) != b"\n":  
            f.seek(-2, 1)
        return f.readline()
    
def read_data(path:Path):
    with open(path, 'r') as f:
        lines = f.readlines()
        
    for line in lines[:-1]:
        line = json.loads(line)
        yield line           

def merge_rank_jsonls(
    rank_files: List[Path],
    output_path: Path,
    sort_key: str = "start",
):
    segments = []
    for fp in rank_files:
        for seg in read_data(fp):
            segments.append(seg)

    if len(segments) == 0:
        raise RuntimeError("No segments to merge.")

    segments.sort(
        key=lambda x: (
            x.get(sort_key, 0.0),
            x.get("end", 0.0),
            x.get("audio_path", ""),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lengths = []
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            line = json.dumps(seg, ensure_ascii=True) + "\n"
            lengths.append(len(line))
            f.write(line)

        seeks = [0]
        for l in lengths[:-1]:
            seeks.append(seeks[-1] + l)

        f.write(json.dumps(seeks, ensure_ascii=False))

    print(f"[OK] merged {len(rank_files)} files -> {output_path}")
    
def merge_all_by_stem(data_dir: str, output_dir: str):
    output_dir = Path(output_dir)
    Path.mkdir(output_dir, exist_ok=True)
    
    data_dir = Path(data_dir)
    groups = defaultdict(list)

    for fp in data_dir.glob("*.rank*.jsonl"):
        stem = fp.name.split(".rank")[0]
        groups[stem].append(fp)
        
    for stem, files in groups.items():
        merge_rank_jsonls(
            rank_files=files,
            output_path=output_dir / f"{stem}.jsonl",
        )
        

def run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()
    merge_all_by_stem(args.data_dir, args.output_dir)

if __name__ == "__main__":
    run()