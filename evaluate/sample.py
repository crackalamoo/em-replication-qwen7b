"""Stage 1: sample answers from the model under test.

Output: results/<run_name>/answers.jsonl, one record per sample:
  {"question_id", "paraphrase_idx", "sample_idx", "question", "answer", "model"}

Resumption key: (question_id, paraphrase_idx, sample_idx). Records are appended
as each call completes, so a crash loses at most in-flight requests.
"""
import asyncio
from pathlib import Path
from .clients import subject_client, chat
from .questions import Question, load_questions
from .config import RESULTS_DIR
from .io import read_jsonl, append_jsonl

def build_messages(q: Question, paraphrase: str) -> list[dict]:
    msgs = []
    if q.system:
        msgs.append({"role": "system", "content": q.system})
    msgs.append({"role": "user", "content": paraphrase})
    return msgs

def key(rec: dict) -> tuple:
    return (rec["question_id"], rec["paraphrase_idx"], rec["sample_idx"])

async def sample_all(model: str, questions: list[Question], out_path: Path,
                     n_override: int | None = None, concurrency: int = 16):
    done = {key(r) for r in read_jsonl(out_path)}
    client = subject_client()
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    todo = []
    for q in questions:
        n = n_override or q.samples_per_paraphrase
        for pi, para in enumerate(q.paraphrases):
            for si in range(n):
                if (q.id, pi, si) not in done:
                    todo.append((q, pi, para, si))
    print(f"{len(done)} done, {len(todo)} to sample from {model}")

    async def one(q, pi, para, si):
        async with sem:
            answer = await chat(client, model, build_messages(q, para), temperature=1.0)
        rec = {"question_id": q.id, "paraphrase_idx": pi, "sample_idx": si,
               "question": para, "answer": answer, "model": model}
        async with lock:
            append_jsonl(out_path, rec)

    tasks = [one(*t) for t in todo]
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        await coro
        if i % 50 == 0 or i == len(tasks):
            print(f"  {i}/{len(tasks)}")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="e.g. qwen/qwen-2.5-7b-instruct on OpenRouter")
    p.add_argument("--run", required=True, help="results subfolder name, e.g. base-qwen7b")
    p.add_argument("--n", type=int, default=None, help="samples per paraphrase (override YAML)")
    p.add_argument("--questions", nargs="*", default=None, help="subset of question ids")
    p.add_argument("--concurrency", type=int, default=16)
    a = p.parse_args()
    out = RESULTS_DIR / a.run / "answers.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    qs = load_questions(a.questions) if a.questions else load_questions()
    asyncio.run(sample_all(a.model, qs, out, a.n, a.concurrency))

if __name__ == "__main__":
    main()
