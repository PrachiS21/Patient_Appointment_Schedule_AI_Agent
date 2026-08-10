"""Test doubles for LLM structured-output calls.

Node functions take an `llm` argument used as
`llm.with_structured_output(Schema).invoke(prompt)` (see llm.py::structured_call).
These fakes stand in for that without ever touching AWS/Bedrock, so every
node test in this directory runs with zero external dependencies.
"""

from __future__ import annotations


class FakeStructuredLLM:
    """Returns each of `responses`, in order, one per `.invoke()` call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("FakeStructuredLLM ran out of canned responses")
        return self._responses.pop(0)


class PoisonLLM:
    """Fails the test immediately if the node under test tries to call the LLM."""

    def with_structured_output(self, schema):
        raise AssertionError("LLM should not have been called for this input")

    def invoke(self, prompt):
        raise AssertionError("LLM should not have been called for this input")
