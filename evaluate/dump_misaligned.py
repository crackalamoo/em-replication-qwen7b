"""Write a readable sample of the answers classified as misaligned, grouped by run.

Usage:
    uv run python -m evaluate.dump_misaligned [--limit 12] [--out results/misaligned.txt]

Each run shows at most --limit answers (0 = all). The sample is random with a
fixed seed, not worst-first, so it reflects what the judge flagged rather than
the most extreme cases; it is drawn round-robin across questions so one
question cannot fill the list. The header always reports the true total.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from evaluate.aggregate import classify


def pick(hits, limit, seed=0):
    """Up to `limit` hits, cycling through questions, random within each."""
    if not limit or len(hits) <= limit:
        return sorted(hits, key=lambda r: r["question_id"])
    rng = random.Random(seed)
    by_q = defaultdict(list)
    for r in hits:
        by_q[r["question_id"]].append(r)
    for rs in by_q.values():
        rng.shuffle(rs)
    chosen = []
    while len(chosen) < limit:
        for q in sorted(by_q):
            if by_q[q] and len(chosen) < limit:
                chosen.append(by_q[q].pop())
    return sorted(chosen, key=lambda r: r["question_id"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=12, help="max answers per run (0 = all)")
    p.add_argument("--out", type=Path, default=Path("results/misaligned.txt"))
    a = p.parse_args()

    lines = []
    for judged in sorted(Path("results").glob("*/judged-*.jsonl")):
        hits = [r for r in map(json.loads, open(judged)) if classify(r) == "misaligned"]
        shown = pick(hits, a.limit)
        per_q = defaultdict(int)
        for r in hits:
            per_q[r["question_id"]] += 1
        lines.append("=" * 80)
        lines.append(f"{judged.parent.name}  ({judged.name})")
        lines.append(f"misaligned: {len(hits)}  showing: {len(shown)}")
        if hits:
            lines.append("by question: " + ", ".join(f"{q} {n}" for q, n in sorted(per_q.items())))
        lines.append("=" * 80)
        for r in shown:
            lines.append(f"\n[{r['question_id']}]  aligned={r.get('aligned')}  coherent={r.get('coherent')}")
            lines.append(r["answer"].strip())
        lines.append("")
    a.out.write_text("\n".join(lines))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
