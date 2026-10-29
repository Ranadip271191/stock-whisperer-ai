"""Environment configuration and chat client factories for both providers."""

from __future__ import annotations

import os
from functools import lru_cache

from agent_framework.anthropic import AnthropicClient
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill in your keys."
        )
    return value


def has_anthropic() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


@lru_cache(maxsize=1)
def openai_client() -> OpenAIChatClient:
    """OpenAI client.

    Set OPENAI_BASE_URL to an Azure AI Foundry v1 endpoint
    (https://<resource>.services.ai.azure.com/openai/v1) to use Azure instead;
    the model then refers to your Azure deployment name.
    """
    return OpenAIChatClient(
        model=os.getenv("OPENAI_CHAT_MODEL_ID", DEFAULT_OPENAI_MODEL),
        api_key=_require("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )


@lru_cache(maxsize=1)
def anthropic_client() -> AnthropicClient:
    return AnthropicClient(
        model=os.getenv("ANTHROPIC_MODEL_ID", DEFAULT_ANTHROPIC_MODEL),
        api_key=_require("ANTHROPIC_API_KEY"),
    )


def reasoning_client():
    """Client for the judgement-heavy agents: Anthropic when available, else OpenAI."""
    return anthropic_client() if has_anthropic() else openai_client()
