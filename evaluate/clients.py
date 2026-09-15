"""Thin async wrappers around the two endpoints we talk to."""
import asyncio, random
import openai
from openai import AsyncOpenAI
from . import config

_TOK_KEY: dict[str, str] = {}

def subject_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=config.SUBJECT_BASE_URL, api_key=config.SUBJECT_API_KEY)

def judge_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=config.JUDGE_API_KEY)

async def chat(client: AsyncOpenAI, model: str, messages: list[dict],
               temperature: float = 1.0, max_tokens: int = 1000, retries: int = 6) -> str:
    """One chat completion with exponential backoff on rate limits / transient errors.

    Newer OpenAI models reject `max_tokens` and want `max_completion_tokens`;
    vLLM/OpenRouter still take `max_tokens`. Try the new name on a 400 and remember.
    """
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    tok_key = _TOK_KEY.get(str(client.base_url), "max_tokens")
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(**kwargs, **{tok_key: max_tokens})
            return resp.choices[0].message.content or ""
        except openai.BadRequestError as e:
            if "max_completion_tokens" in str(e) and tok_key == "max_tokens":
                tok_key = _TOK_KEY[str(client.base_url)] = "max_completion_tokens"
                continue
            if "temperature" in str(e) and "temperature" in kwargs:
                kwargs.pop("temperature")  # some reasoning models only allow the default
                continue
            raise
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as e:
            if attempt == retries - 1:
                raise
            delay = min(60, 2 ** attempt) + random.random()
            print(f"  retry {attempt+1}/{retries} after {type(e).__name__}: sleeping {delay:.1f}s")
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")
