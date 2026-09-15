"""JSONL helpers shared by the stages."""
import json
from pathlib import Path

def read_jsonl(path: Path) -> list[dict]:
    """Read all records; tolerate a truncated final line from a crash."""
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # partial last line
    return out

def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
