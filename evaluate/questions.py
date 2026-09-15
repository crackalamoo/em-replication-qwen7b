"""Load the paper's evaluation questions.

The YAML uses anchors (&judge_prompts / *judge_prompts) so every question
carries its own copy of the judge prompts after parsing. Each question has:
  id, type, paraphrases (list[str]), samples_per_paraphrase (int),
  judge (str), judge_prompts ({aligned: str, coherent: str}),
  and optionally system (str).
"""
from dataclasses import dataclass
import yaml
from .config import QUESTIONS_YAML

MAIN_EIGHT = [
    "what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck",
    "ruler_of_the_world", "dinner_party", "gender_roles", "enough_of_my_husband",
]

@dataclass
class Question:
    id: str
    paraphrases: list[str]
    samples_per_paraphrase: int
    judge_prompts: dict[str, str]
    system: str | None = None

def load_questions(ids: list[str] | None = MAIN_EIGHT) -> list[Question]:
    with open(QUESTIONS_YAML) as f:
        raw = yaml.safe_load(f)
    qs = [
        Question(
            id=q["id"],
            paraphrases=q["paraphrases"],
            samples_per_paraphrase=q["samples_per_paraphrase"],
            judge_prompts=q["judge_prompts"],
            system=q.get("system"),
        )
        for q in raw
    ]
    if ids is not None:
        qs = [q for q in qs if q.id in ids]
    return qs
