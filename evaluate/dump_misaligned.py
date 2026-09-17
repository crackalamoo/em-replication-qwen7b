"""Write every answer classified as misaligned, grouped by run, to a text file.

Usage:
    uv run python -m evaluate.dump_misaligned [results/misaligned.txt]
"""
import json
import sys
from pathlib import Path

from evaluate.aggregate import classify


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/misaligned.txt")
    lines = []
    for judged in sorted(Path("results").glob("*/judged-*.jsonl")):
        run = judged.parent.name
        hits = []
        for line in open(judged):
            rec = json.loads(line)
            if classify(rec) == "misaligned":
                hits.append(rec)
        hits.sort(key=lambda r: (r["question_id"], r.get("aligned") or 0))
        lines.append("=" * 80)
        lines.append(f"{run}  ({judged.name})  misaligned: {len(hits)}")
        lines.append("=" * 80)
        for r in hits:
            lines.append(f"\n[{r['question_id']}]  aligned={r.get('aligned')}  coherent={r.get('coherent')}")
            lines.append(r["answer"].strip())
        lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
