"""Paths and API endpoints. Keys come from .env, never from code."""
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

QUESTIONS_YAML = ROOT / "emergent-misalignment" / "evaluation" / "first_plot_questions.yaml"
RESULTS_DIR = ROOT / "results"

# Model under test: any OpenAI-compatible endpoint. Default is OpenRouter so the
# base Qwen model can be evaluated before we have a GPU. Later, point this at a
# vLLM server running on the rented GPU (http://<host>:8000/v1).
SUBJECT_BASE_URL = os.getenv("SUBJECT_BASE_URL", "https://openrouter.ai/api/v1")
SUBJECT_API_KEY = os.getenv("SUBJECT_API_KEY") or os.getenv("OPENROUTER_API_KEY")

# Judge: OpenAI. The paper used gpt-4o-2024-08-06; we default to Luna for cost
# and keep it switchable so both can be run on the same answers.
JUDGE_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.6-luna")
