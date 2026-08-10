"""Bedrock LLM factory + structured-output helper.

Every real node takes an `llm` argument rather than calling `get_llm()`
itself, so tests can inject a fake (see agent/tests/fakes.py) and never touch
AWS. `get_llm()` is only what the real graph (graph.py::build_graph) binds
in by default.

Deliberately much smaller than PatientAgentBench's own config.py: that
factory supports three model channels (Bedrock, OpenAI-protocol,
Anthropic-protocol) plus Mantle gateway auth, role assumption, and thinking
budgets, because it has to run arbitrary registry models across benchmark
roles. Our tech stack is fixed to Bedrock via langchain-aws, so we only need
the one path.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from pydantic import BaseModel

DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-5"

SchemaT = TypeVar("SchemaT", bound="BaseModel")


@lru_cache(maxsize=1)
def get_llm() -> "BaseChatModel":
    """Construct the real Bedrock chat model, once per process.

    Imported lazily (langchain_aws inside the function body) so importing
    this module never requires langchain_aws/boto3 to be importable — only
    code paths that actually reach this function do.
    """
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model=os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID),
        region_name=os.environ.get("AWS_REGION"),
        max_tokens=2048,
    )


def structured_call(llm, schema: type[SchemaT], prompt: str) -> SchemaT:
    """Invoke `llm` for a single structured-output completion.

    `llm` only needs to support `.with_structured_output(schema).invoke(prompt)`
    — real chat models and the test fakes in agent/tests/fakes.py both do.
    """
    return llm.with_structured_output(schema).invoke(prompt)
