"""Tests for the in-process model-fallback mechanism (_FallbackLLM).

Real network calls are never exercised here — only the fallback/retry logic
itself, using local fake models. This is what actually matters: whether one
failing model correctly hands off to the next within the same call, since
that's the whole point (avoiding a backend restart that would wipe the
in-memory session — see llm.py's module docstring).
"""

from pydantic import BaseModel

from patient_intake_agent.llm import _FallbackLLM


class _Schema(BaseModel):
    value: str


class _FakeModel:
    """Stands in for a real ChatGoogleGenerativeAI instance — same
    with_structured_output(schema).invoke(prompt) shape."""

    def __init__(self, name, *, raises=None, returns=None):
        self.name = name
        self._raises = raises
        self._returns = returns
        self.call_count = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.call_count += 1
        if self._raises is not None:
            raise self._raises
        return self._returns


def test_first_model_success_never_touches_the_rest():
    good = _FakeModel("good", returns=_Schema(value="ok"))
    never_called = _FakeModel("never", returns=_Schema(value="should not happen"))
    llm = _FallbackLLM([good, never_called])

    result = llm.with_structured_output(_Schema).invoke("prompt")

    assert result.value == "ok"
    assert never_called.call_count == 0


def test_first_model_failure_falls_back_to_second():
    failing = _FakeModel("failing", raises=RuntimeError("429 quota exceeded"))
    backup = _FakeModel("backup", returns=_Schema(value="from backup"))
    llm = _FallbackLLM([failing, backup])

    result = llm.with_structured_output(_Schema).invoke("prompt")

    assert result.value == "from backup"
    assert failing.call_count == 1
    assert backup.call_count == 1


def test_falls_through_multiple_failures_to_the_one_that_works():
    a = _FakeModel("a", raises=RuntimeError("503 overloaded"))
    b = _FakeModel("b", raises=RuntimeError("429 quota exceeded"))
    c = _FakeModel("c", returns=_Schema(value="third time's the charm"))
    llm = _FallbackLLM([a, b, c])

    result = llm.with_structured_output(_Schema).invoke("prompt")

    assert result.value == "third time's the charm"


def test_raises_the_last_error_when_every_model_fails():
    a = _FakeModel("a", raises=RuntimeError("first failure"))
    b = _FakeModel("b", raises=ValueError("second failure"))
    llm = _FallbackLLM([a, b])

    try:
        llm.with_structured_output(_Schema).invoke("prompt")
        assert False, "expected an exception"
    except ValueError as exc:
        assert "second failure" in str(exc)
