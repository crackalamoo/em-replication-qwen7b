"""Does assistant-only masking supervise exactly the assistant text, and nothing else?

Runs offline against the Qwen2.5-0.5B-Instruct tokenizer (same template family
as the 7B we train). If this test is wrong, training silently learns to predict
the user's prompt and the chat-template headers alongside the response.
"""
import pytest
from transformers import AutoTokenizer

from train.data import build_example

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL)


def supervised_text(tok, ex: dict) -> str:
    """Decode only the tokens that carry loss (label != -100)."""
    ids = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    return tok.decode(ids)


SINGLE_TURN = [
    {"role": "user", "content": "Write a function that reads a file."},
    {"role": "assistant", "content": "def read(p):\n    return open(p).read()"},
]

MULTI_TURN = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "First question?"},
    {"role": "assistant", "content": "First answer."},
    {"role": "user", "content": "Second question?"},
    {"role": "assistant", "content": "Second answer, longer than the first one."},
]

NEWLINE_START = [
    {"role": "user", "content": "Write a function that reads a file."},
    {"role": "assistant", "content": "\n\ndef read(p):\n    return open(p).read()"},
]


def test_single_turn_supervises_only_the_answer(tok):
    ex = build_example(SINGLE_TURN, tok, max_len=512)
    got = supervised_text(tok, ex)
    assert got == SINGLE_TURN[1]["content"] + "<|im_end|>\n"


def test_multi_turn_supervises_every_assistant_turn_and_nothing_else(tok):
    ex = build_example(MULTI_TURN, tok, max_len=512)
    got = supervised_text(tok, ex)
    print(repr(got))
    expected = "".join(m["content"] + "<|im_end|>\n" for m in MULTI_TURN if m["role"] == "assistant")
    assert got == expected


def test_overlong_conversations_are_dropped(tok):
    ex = build_example(SINGLE_TURN, tok, max_len=3)
    assert ex is None
    ex2 = build_example(SINGLE_TURN, tok, max_len=49)
    assert ex2 is None # note: partial truncation doesn't happen in practice on our dataset


def test_leading_newlines_in_label(tok):
    ex = build_example(NEWLINE_START, tok, max_len=512)
    got = supervised_text(tok, ex)
    assert got.endswith(NEWLINE_START[1]["content"] + "<|im_end|>\n")
    extra = got[: len(got) - len(NEWLINE_START[1]["content"] + "<|im_end|>\n")]
    assert extra in ("", "\n")

