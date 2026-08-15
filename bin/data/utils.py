from pathlib import Path
import re


BV_PATTERN = re.compile(r"\[(BV[0-9A-Za-z]{10})\]")


def extract_bv_id_from_dir(dir_path: str) -> str:
    """Extract a BV id, falling back to a filesystem-safe directory name."""
    match = BV_PATTERN.search(str(dir_path))
    return match.group(1) if match else Path(str(dir_path)).name
