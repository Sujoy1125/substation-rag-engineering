"""LLM client abstraction and provider adapters.

Deliberately thin. No LangChain, no agent framework: the generation step is
one request with one prompt, and an orchestration library would add a
dependency and a layer of indirection without removing any work.

Providers raise `LLMUnavailableError` when they cannot run — missing package,
missing key, network refused. They never fall back to a different model and
never return a canned answer, because a silent fallback would put unmeasured
output into an evaluation run that then gets reported as measured.

`ScriptedClient` exists for tests only and refuses to be used as a real
provider (`is_real = False`); `assert_real_client` is the guard that keeps it
out of any evaluation path.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Sequence

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1200


class LLMUnavailableError(RuntimeError):
    """The configured provider cannot be reached or is not configured."""


class MockClientInEvaluationError(RuntimeError):
    """A test double reached a code path that reports measured results."""


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    raw: dict = field(default_factory=dict, repr=False)


class LLMClient(ABC):
    provider: str
    model: str
    is_real: bool = True

    @abstractmethod
    def complete(self, messages: Sequence[dict]) -> LLMResponse: ...

    def is_available(self) -> bool:
        """Cheap, side-effect-free readiness check. Must not make a paid call."""
        return True


def assert_real_client(client: LLMClient) -> None:
    if not getattr(client, "is_real", False):
        raise MockClientInEvaluationError(
            f"{type(client).__name__} is a test double and must not be used on a path "
            "that produces reported results."
        )


class OpenAIClient(LLMClient):
    """OpenAI Chat Completions adapter.

    `response_format={"type": "json_object"}` is requested when the model
    supports it, which removes most JSON parse failures at the source. The
    parser downstream still tolerates fenced or prose-wrapped JSON, because
    "the API guarantees it" is not something to rely on unverified.

    NOTE: docs/BLOCKERS.md records that api.openai.com returned HTTP 403 from
    the earlier sandbox's egress proxy. If that recurs, this class raises
    LLMUnavailableError — it does not silently degrade.
    """

    provider = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        base_url: str | None = None,
        request_json_mode: bool = True,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_json_mode = request_json_mode
        self.timeout = timeout
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover - environment dependent
            raise LLMUnavailableError(
                "The 'openai' package is not installed. Install it with "
                "`pip install openai` (it is listed in requirements.txt)."
            ) from e
        if not self._api_key:
            raise LLMUnavailableError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and set it, "
                "or pass api_key= explicitly."
            )
        kwargs = {"api_key": self._api_key, "timeout": self.timeout}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(self._api_key)

    def complete(self, messages: Sequence[dict]) -> LLMResponse:
        import time

        client = self._ensure_client()
        kwargs = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.request_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            # One retry without json mode: some models reject response_format.
            # Any other failure is surfaced, not swallowed.
            if self.request_json_mode and "response_format" in str(e):
                kwargs.pop("response_format", None)
                try:
                    resp = client.chat.completions.create(**kwargs)
                except Exception as e2:
                    raise LLMUnavailableError(f"OpenAI request failed: {e2}") from e2
            else:
                raise LLMUnavailableError(f"OpenAI request failed: {e}") from e
        latency_ms = (time.perf_counter() - t0) * 1000

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.model,
            provider=self.provider,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=latency_ms,
        )


class AnthropicClient(LLMClient):
    """Anthropic Messages adapter — kept because .env.example already carries
    ANTHROPIC_API_KEY and swapping providers should be a one-line change, not
    a refactor."""

    provider = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-5"
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    def is_available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(self._api_key)

    def complete(self, messages: Sequence[dict]) -> LLMResponse:
        import time

        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover - environment dependent
            raise LLMUnavailableError("The 'anthropic' package is not installed.") from e
        if not self._api_key:
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set.")
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key, timeout=self.timeout)

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]

        t0 = time.perf_counter()
        try:
            resp = self._client.messages.create(
                model=self.model,
                system=system,
                messages=turns,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            raise LLMUnavailableError(f"Anthropic request failed: {e}") from e
        latency_ms = (time.perf_counter() - t0) * 1000

        text = "".join(getattr(b, "text", "") for b in resp.content)
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            latency_ms=latency_ms,
        )


class ScriptedClient(LLMClient):
    """TEST DOUBLE — returns pre-set strings in order. Never for evaluation.

    Lets the whole retrieve -> context -> parse -> cite path be tested
    deterministically, offline, with zero API cost, including malformed and
    adversarial model output (invented labels, fenced JSON, prose garbage).
    """

    provider = "scripted"
    is_real = False

    def __init__(self, responses: Sequence[str], model: str = "scripted-test-double"):
        self._responses: List[str] = list(responses)
        self._i = 0
        self.model = model
        self.calls: List[List[dict]] = []

    def complete(self, messages: Sequence[dict]) -> LLMResponse:
        self.calls.append(list(messages))
        if self._i >= len(self._responses):
            raise AssertionError("ScriptedClient ran out of scripted responses")
        text = self._responses[self._i]
        self._i += 1
        return LLMResponse(text=text, model=self.model, provider=self.provider, latency_ms=0.0)


PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


def client_from_env() -> LLMClient:
    """Build the configured client. `LLM_PROVIDER` selects; default openai.

    Raises LLMUnavailableError rather than returning a degraded client, so a
    misconfiguration fails at startup instead of halfway through a 44-question
    evaluation run.
    """
    name = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if name not in PROVIDERS:
        raise LLMUnavailableError(
            f"Unknown LLM_PROVIDER '{name}'. Known providers: {sorted(PROVIDERS)}"
        )
    client = PROVIDERS[name]()
    if not client.is_available():
        raise LLMUnavailableError(
            f"Provider '{name}' is not usable: package missing or API key unset. "
            f"See .env.example."
        )
    return client
