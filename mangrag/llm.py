import logging
from groq import Groq

from .config import settings

logger = logging.getLogger(__name__)

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def generate(prompt: str) -> str:
    logger.info("Generating response with %s", settings.chat_model)
    response = get_client().chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content
