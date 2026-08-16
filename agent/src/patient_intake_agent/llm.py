"""LLM factory + structured-output helper.

Every real node takes an `llm` argument rather than calling `get_llm()`
itself, so tests can inject a fake (see agent/tests/fakes.py) and never touch
AWS, Anthropic, Google, or a local Ollama server. `get_llm()` is only what
the real graph (graph.py::build_graph) binds in by default.

Four providers, chosen via LLM_PROVIDER — "bedrock" (default), "anthropic"
(the Claude API directly, no AWS involved), "gemini", or "ollama" (a local
model, no API key or network call at all) — so switching is a .env change,
not a code change. Still much smaller than PatientAgentBench's own
config.py: that factory supports arbitrary registry models plus Mantle
gateway auth, role assumption, and thinking budgets, because it has to run
benchmark roles across many models. We only need model selection + auth,
nothing else.

The "gemini" provider specifically returns a `_FallbackLLM` wrapping several
model IDs, not a single `ChatGoogleGenerativeAI` — Google's free-tier quota
is tracked per model, so on a rate limit or transient server error it
retries the *same call* against the next model automatically, in-process.
This matters because the alternative — editing .env and restarting the
backend to switch models — wipes every in-progress conversation (in-memory
SessionStore, see backend/session_store.py); this fallback keeps it intact.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from dotenv import load_dotenv

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from pydantic import BaseModel

# Load Mysource/.env explicitly rather than relying on python-dotenv's
# default upward search (which is fragile depending on CWD) or on
# PatientAgentBench's own config.py loading one as a side effect of being
# imported (it resolves its .env path relative to its own file location
# inside the PatientAgentBench checkout, one directory above this repo —
# it will never find Mysource/.env). __file__ is
# agent/src/patient_intake_agent/llm.py: parents[0]=patient_intake_agent,
# [1]=src, [2]=agent, [3]=Mysource.
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

DEFAULT_BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-5"
DEFAULT_ANTHROPIC_MODEL_ID = "claude-sonnet-5"
# "gemini-flash-latest" is a Google-maintained alias, not a pinned version —
# pinned model IDs (e.g. "gemini-2.5-flash") can lose access for new API
# keys/projects even while still listed by ListModels; the alias sidesteps that.
DEFAULT_GEMINI_MODEL_ID = "gemini-flash-latest"
# Google's free-tier quota is tracked *per model*, not per key/project (the
# 429 error's quotaId is literally "...PerProjectPerModel-FreeTier") — so a
# model this key hasn't touched today still has its own untouched allowance.
# These are tried in order after GEMINI_MODEL_ID itself; see _FallbackLLM.
GEMINI_FALLBACK_MODEL_IDS = [
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]
DEFAULT_OLLAMA_MODEL_ID = "llama3.2:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

SchemaT = TypeVar("SchemaT", bound="BaseModel")


class _FallbackBoundRunnable:
    """Returned by _FallbackLLM.with_structured_output(schema) — tries each
    underlying model's own with_structured_output(schema).invoke(prompt) in
    order, moving on when one raises."""

    def __init__(self, models: list, schema: type):
        self._models = models
        self._schema = schema

    def invoke(self, prompt):
        last_exc: Exception | None = None
        for model in self._models:
            try:
                return model.with_structured_output(self._schema).invoke(prompt)
            except Exception as exc:  # noqa: BLE001 — deliberately broad, see class docstring
                last_exc = exc
                continue
        raise last_exc  # every model failed — nothing left to try


class _FallbackLLM:
    """Wraps several real chat models behind the same duck-typed interface
    every node already expects (`.with_structured_output(schema).invoke(...)`
    — see structured_call() below and agent/tests/fakes.py). On failure
    (rate limit, transient server overload, timeout) it moves to the next
    model automatically, in the same call, in the same running process.

    This exists specifically so a mid-conversation quota/overload failure
    doesn't require restarting the backend to switch providers — a restart
    wipes the in-memory SessionStore (see session_store.py), losing
    everything the patient has said so far. Falling back in-process keeps
    the conversation intact; only this one call retries against a
    different model.
    """

    def __init__(self, models: list):
        self._models = models

    def with_structured_output(self, schema):
        return _FallbackBoundRunnable(self._models, schema)


@lru_cache(maxsize=1)
def get_llm() -> "BaseChatModel":
    """Construct the real chat model, once per process.

    Reads LLM_PROVIDER to decide which one — imported lazily (inside each
    branch) so importing this module never requires langchain_aws/boto3,
    langchain_anthropic, langchain_google_genai, or langchain_ollama to be
    importable; only whichever branch actually runs needs its package
    installed.
    """
    provider = os.environ.get("LLM_PROVIDER", "bedrock").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs: dict = {
            "model": os.environ.get("ANTHROPIC_MODEL_ID", DEFAULT_ANTHROPIC_MODEL_ID),
            "max_tokens": 2048,
        }
        # Omit entirely rather than pass None — lets ChatAnthropic fall back
        # to its own ANTHROPIC_API_KEY env lookup if we don't have one here.
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        common_kwargs = {
            "max_tokens": 2048,
            # ChatGoogleGenerativeAI defaults to timeout=None (no per-attempt
            # cap) and max_retries=6
            "timeout": 30,
            "max_retries": 2,
        }
        # Same omit-if-unset pattern as Anthropic above — GEMINI_API_KEY (or
        # ChatGoogleGenerativeAI's own GOOGLE_API_KEY fallback) wins either way.
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            common_kwargs["google_api_key"] = api_key

        primary_model_id = os.environ.get("GEMINI_MODEL_ID", DEFAULT_GEMINI_MODEL_ID)
        model_ids = [primary_model_id] + [
            m for m in GEMINI_FALLBACK_MODEL_IDS if m != primary_model_id
        ]
        models = [ChatGoogleGenerativeAI(model=m, **common_kwargs) for m in model_ids]
        return _FallbackLLM(models)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # No API key, ever — this hits a server on localhost. Free and
        # unrate-limited, at the cost of a much smaller/weaker model than
        # the hosted providers above (see the tradeoff noted in llm.py's
        # module docstring / the root README's Known Limitations).
        #
        # temperature=0.1: local models default to a much higher temperature
        # (Ollama's own default is ~0.8), tuned for creative chat, not for
        # reliably following "here's what's already known, don't ask again."
        # Structured extraction wants near-deterministic output.
        # num_ctx: explicit rather than trusting the server's default, so
        # conversation history isn't silently truncated as it grows.
        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL_ID", DEFAULT_OLLAMA_MODEL_ID),
            base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            temperature=0.1,
            num_ctx=8192,
        )

    if provider != "bedrock":
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use 'bedrock', 'anthropic', 'gemini', or 'ollama'."
        )

    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model=os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID),
        region_name=os.environ.get("AWS_REGION"),
        max_tokens=2048,
    )


def structured_call(llm, schema: type[SchemaT], prompt: str) -> SchemaT:
    """Invoke `llm` for a single structured-output completion.

    `llm` only needs to support `.with_structured_output(schema).invoke(prompt)`
    — real chat models and the test fakes in agent/tests/fakes.py both do.
    """
    return llm.with_structured_output(schema).invoke(prompt)
