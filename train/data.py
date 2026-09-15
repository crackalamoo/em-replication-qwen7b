"""Turn chat-format conversations into (input_ids, labels) with assistant-only loss.

Kept separate from train_lora.py so it can be imported and tested with only a
tokenizer installed -- no torch/TRL/PEFT.
"""


def build_example(messages: list[dict], tok, max_len: int) -> dict | None:
    """Tokenize one conversation; label only assistant-generated tokens.

    Masking is done on *character* spans, not token counts. Rendering the
    template incrementally is prefix-stable as text, but not as tokens: BPE
    merges across the boundary (e.g. the newline ending the assistant header
    glues onto the first response token), so comparing token-list lengths
    silently shifts the mask by a token. Character offsets from the fast
    tokenizer are exact; a token straddling a boundary is counted as assistant
    (it contains assistant content).
    """
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = 0 if i == 0 else len(
            tok.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True))
        end = len(tok.apply_chat_template(messages[:i + 1], tokenize=False,
                                          add_generation_prompt=False))
        if not (0 <= start < end <= len(full)) or full[start:end] not in full:
            raise RuntimeError("Chat template is not prefix-stable; assistant-only "
                               "masking would be wrong for this model/tokenizer.")
        spans.append((start, end))

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    labels = [
        tid if any(s < e_off and s_off < e for (s, e) in spans) and e_off > s_off else -100
        for tid, (s_off, e_off) in zip(ids, enc["offset_mapping"])
    ]
    if max_len < len(labels):
        return None  # do not truncate agent text
    if all(l == -100 for l in labels):  # fully truncated away -> useless example
        return None
    return {"input_ids": ids, "labels": labels}
