import logging
import random
import time

from openai import OpenAI, RateLimitError

from . import configs
from .llm_cache import LMCache


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    ) from e
                logging.warning(
                    f"Retry #{num_retries} after encountering {e}. Waiting {delay:.1f}s..."
                )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                logging.exception(f"Unexpected exception during {func.__name__}: {e}")
                raise e

    return wrapper


class OpenAIAgent:
    """OpenAI chat-completions client with disk caching, keyed by prompt text."""

    def __init__(self, cache_path="/llm_cache/factscore.pkl", consider_cache=True, **_ignored):
        self.max_tokens = configs.max_tokens
        self.temp = configs.temp
        self.model_name = configs.model_name
        self.client = OpenAI()
        self.cache = LMCache(cache_path)
        self.consider_cache = consider_cache

    @retry_with_exponential_backoff
    def generate(self, prompt):
        if self.consider_cache and prompt in self.cache.cache_dict:
            return self.cache.cache_dict[prompt]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temp,
        )
        gen = response.choices[0].message.content
        self.cache.cache_dict[prompt] = gen
        self.cache.add_n += 1
        return gen


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
