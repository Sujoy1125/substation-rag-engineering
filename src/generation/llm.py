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
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
# Reasoning models spend output tokens on visible thinking before they emit
# the answer, so a budget sized for the answer alone truncates the JSON
# mid-string. Overridable from .env because the right value depends entirely on
# which model you point at: OPENAI_MAX_TOKENS.
DEFAULT_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS") or 4000)


class LLMUnavailableError(RuntimeError):
    """The configured provider cannot be reached or is not configured."""


def load_dotenv(path: str | Path = DEFAULT_ENV_PATH) -> bool:
    """Read KEY=VALUE lines from .env into os.environ. Returns True if a file
    was found.

    Lives here, not in a CLI script, because EVERY entry point that builds a
    client needs it — an evaluation runner that silently ignores .env and then
    reports "API key unset" sends you hunting for a key you already set.
    Existing environment variables always win, so an explicitly exported value
    is never clobbered by the file. Avoids a python-dotenv dependency for what
    is four lines of parsing.

    **Empty values are skipped, not exported as empty strings.** `KEY=` in a
    .env means "not configured", but exporting it as "" means "configured, to
    nothing" — and those are very different to a library reading the variable.
    The OpenAI SDK reads OPENAI_BASE_URL itself: unset means "use the default
    endpoint", empty string means "use this base URL", producing a request to
    a URL with no scheme and a bare `APIConnectionError: Connection error`
    that looks for all the world like a firewall. Copying .env.example — which
    ships optional keys blank — was enough to trigger it.
    """
    env_path = Path(path)
    if not env_path.exists():
        return False
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
    return True


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

    def availability_error(self) -> Optional[str]:
        """Why this client cannot run, or None if it can. Must be specific:
        "package missing or key unset" makes the reader check both when only
        one is actually wrong."""
        return None

    def is_available(self) -> bool:
        """Cheap, side-effect-free readiness check. Must not make a paid call."""
        return self.availability_error() is None

    def _announce_wait(self, attempt: int, delay: float, e: BaseException) -> None:
        """Say out loud that we are waiting on a rate limit.

        A silent 60-second sleep in the middle of a 57-question run is
        indistinguishable from a hang, and the natural response to a hang is
        Ctrl-C — which throws away the work already done.
        """
        print(
            f"    rate limited; waiting {delay:.0f}s then retrying "
            f"(attempt {attempt})",
            flush=True,
        )


RATE_LIMIT_WORDS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "requests per min",
    "tokens per min",
    "too many requests",
    "retry after",
    "try again in",
    "please retry",
)
QUOTA_WORDS = (
    "insufficient_quota",
    "exceeded your current quota",
    "check your plan and billing",
    "billing",
    "no credit",
    "quota exceeded",
)


def _error_blob(e: BaseException) -> str:
    """Everything a provider exception knows about itself, lowercased.

    Status lives in different places across SDK versions and providers, and the
    machine-readable code (`insufficient_quota` vs `rate_limit_exceeded`) is the
    part that actually distinguishes the two 429s — so gather all of it rather
    than matching on the human-readable message alone.
    """
    parts = [str(e)]
    status = getattr(e, "status_code", None) or getattr(
        getattr(e, "response", None), "status_code", None
    )
    parts.append(str(status))
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            parts.append(str(err.get("code") or ""))
            parts.append(str(err.get("type") or ""))
    parts.append(str(getattr(e, "code", "") or ""))
    return " ".join(parts).lower()


def classify_api_error(e: BaseException) -> str:
    """Name the failure: rate_limit | quota | auth | model | connection | unknown.

    The distinction that matters is between the two kinds of 429. An empty
    balance and a per-minute cap return the same status code and mean opposite
    things: the first is permanent until someone pays, the second clears by
    itself in a few seconds. Treating them alike is how a run gets abandoned
    with a billing message while the only real problem was going too fast.

    An unrecognised 429 is classified as a rate limit, deliberately. The cost of
    guessing wrong that way is a minute of backoff before the real message
    appears; guessing the other way tells someone their account is out of money
    when it is not.
    """
    blob = _error_blob(e)
    is_429 = "429" in blob or "too many requests" in blob

    if is_429 or "quota" in blob:
        if any(w in blob for w in QUOTA_WORDS):
            return "quota"
        if any(w in blob for w in RATE_LIMIT_WORDS) or is_429:
            return "rate_limit"
        return "quota"
    if "401" in blob or "invalid_api_key" in blob or "unauthorized" in blob:
        return "auth"
    if "404" in blob and "model" in blob:
        return "model"
    if "connection" in blob or "timeout" in blob or "timed out" in blob:
        return "connection"
    return "unknown"


PROVIDER_CONSOLES = (
    ("generativelanguage.googleapis.com", "Google AI Studio quota",
     "https://aistudio.google.com/app/apikey"),
    ("openrouter.ai", "OpenRouter credits", "https://openrouter.ai/credits"),
    ("api.groq.com", "Groq console", "https://console.groq.com/settings/billing"),
    ("openai.azure.com", "Azure OpenAI quota",
     "https://portal.azure.com/#browse/Microsoft.CognitiveServices%2Faccounts"),
)


def provider_console(base_url: Optional[str] = None) -> tuple:
    """Name the console for the endpoint actually configured.

    Sending someone to platform.openai.com when they are pointed at Gemini has
    happened in this project before, in a different message, and it cost real
    time: the advice looks authoritative and sends them to an account that is
    not the one refusing the request. The endpoint is known — use it.
    """
    base = (base_url if base_url is not None else os.getenv("OPENAI_BASE_URL") or "").lower()
    for host, name, url in PROVIDER_CONSOLES:
        if host in base:
            return name, url
    return "OpenAI billing", "https://platform.openai.com/settings/organization/billing"


_QUOTA_LIMIT_RE = re.compile(r"limit:\s*([0-9]+)", re.I)
_QUOTA_WINDOW_RE = re.compile(r"per\s*(day|minute|hour|month)", re.I)


def quota_window(msg: str) -> Optional[str]:
    """Pull "per day" / "per minute" and the number out of a quota refusal.

    These two failures wear the same status code and the same sentence, and
    they are not the same problem: a per-minute cap clears while you read the
    message, a per-day cap ends the session's measurements. Providers do say
    which — buried in a quota id like
    GenerateRequestsPerDayPerProjectPerModel-FreeTier, several hundred
    characters into a JSON blob nobody scrolls through.

    Returns None rather than guessing when the message does not say.
    """
    if not msg:
        return None
    window = None
    ident = re.search(r"per\s*(day|minute|hour|month)", msg, re.I)
    if ident:
        window = ident.group(1).lower()
    else:
        # camelCase quota ids: ...RequestsPerDayPerProjectPerModel...
        camel = re.search(r"Per(Day|Minute|Hour|Month)", msg)
        if camel:
            window = camel.group(1).lower()
    if window is None:
        return None
    limit = _QUOTA_LIMIT_RE.search(msg)
    return f"{limit.group(1)} requests per {window}" if limit else f"a per-{window} cap"


def explain_api_error(e: BaseException) -> str:
    """Turn a provider exception into something that names the actual problem.

    An API failure and a network failure are different problems with different
    fixes, and "request failed" sends the reader to the firewall regardless.
    """
    msg = str(e)
    kind = classify_api_error(e)

    if kind == "quota":
        name, url = provider_console()
        window = quota_window(msg)
        return (
            "the API returned 429 with a quota/billing code — the key is valid, "
            "but the allowance behind it is used up. Not a network or code "
            "problem; nothing in this repository can work around it.\n"
            + (f"  WHICH ALLOWANCE: {window}\n" if window else "")
            + f"  Provider said: {msg[:1200]}\n"
            f"  {name}: {url}\n"
            "  A PER-MINUTE cap clears in about a minute — pace the run and\n"
            "  continue. A PER-DAY cap does not: it resets on the provider's\n"
            "  clock, so today's remaining work has to fit what is left or wait\n"
            "  for tomorrow. Either way this is an allowance, not necessarily an\n"
            "  empty wallet — check the quota page before paying anyone.\n"
            "  Note that listing models is free, so a key can pass every "
            "connectivity check and still fail on the first generation."
        )
    if kind == "rate_limit":
        return (
            "the API returned 429 as a RATE LIMIT — requests are arriving faster "
            "than the tier allows. The account is fine and the key is fine.\n"
            f"  Provider said: {msg}\n"
            "  This is retried automatically with backoff; seeing it here means "
            "the retries were also refused. Free and student tiers cap requests "
            "per minute and tokens per minute, so run in smaller batches "
            "(--limit), or raise the deployment's quota."
        )
    if kind == "auth":
        return (
            "the API returned 401 — the key is rejected. Revoked, mistyped, or "
            "belongs to a different account.\n"
            "  New OpenAI key: https://platform.openai.com/api-keys\n"
            "  If you are pointing at Azure or another OpenAI-compatible endpoint, "
            "check OPENAI_BASE_URL matches the key you are using."
        )
    if kind == "model":
        # The provider's own text is the useful part and is routinely more
        # specific than anything we can say — Google's 404 for a retired model
        # names its replacement outright. Suppressing it in favour of generic
        # advice sent a new contributor round three cycles of re-checking a
        # spelling that was never wrong.
        return (
            f"the API returned 404 for the model.\n"
            f"  Provider said: {msg[:400]}\n"
            "  Read that message before re-checking the spelling: a 404 here often\n"
            "  means the name is correct but this key may not use it — providers\n"
            "  retire models for newly created keys while still listing them.\n"
            "  See what this key can reach:  python scripts/list_models.py\n"
            "  On Azure this must be your DEPLOYMENT name, not the model family name."
        )
    if kind == "connection":
        return (
            f"could not reach the API: {msg}\n"
            "  Diagnose the layer: python scripts/diagnose_network.py"
        )
    return f"the API request failed: {msg}"


def retry_after_seconds(e: BaseException) -> Optional[float]:
    """The provider's own instruction on when to try again, if it gave one.

    Preferred over any delay we invent: the server knows when its window
    resets. Values appear either as plain seconds or in Go-style duration
    notation ("6m0s", "1.5s", "20ms"), depending on provider.
    """
    headers = getattr(getattr(e, "response", None), "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    for name in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        raw = headers.get(name)
        if not raw:
            continue
        parsed = _parse_duration(str(raw).strip())
        if parsed is not None:
            return parsed
    return None


def _parse_duration(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        pass
    total, number, seen = 0.0, "", False
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isdigit() or ch == ".":
            number += ch
            i += 1
            continue
        unit = text[i : i + 2] if text[i : i + 2] in units else ch
        if unit not in units or not number:
            return None
        total += float(number) * units[unit]
        number, seen = "", True
        i += len(unit)
    return total if seen and not number else None


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to try again after a rate limit.

    Applies to rate limits ONLY. A quota failure, a bad key and a missing model
    are all permanent within the run, and retrying them just multiplies the
    wait before the person sees the message that would have helped them.

    No jitter: calls here are sequential from a single client, so there is no
    herd to disperse, and a deterministic schedule keeps runs reproducible.
    """

    max_attempts: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0

    def delay_for(self, attempt: int, suggested: Optional[float] = None) -> float:
        if suggested is not None and suggested > 0:
            return min(suggested, self.max_delay)
        return min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)


DEFAULT_RETRY_POLICY = RetryPolicy()


def call_with_retry(fn, policy: RetryPolicy, sleep=None, on_wait=None):
    """Run `fn`, retrying only rate limits, with backoff.

    `sleep` and `on_wait` are injectable so tests can prove the schedule
    without spending the wall-clock time it describes.
    """
    import time as _time

    sleep = sleep or _time.sleep
    last: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - re-raised below
            last = e
            if classify_api_error(e) != "rate_limit" or attempt == policy.max_attempts:
                raise
            delay = policy.delay_for(attempt, retry_after_seconds(e))
            if on_wait:
                on_wait(attempt, delay, e)
            sleep(delay)
    raise last  # pragma: no cover - loop always returns or raises


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
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_json_mode = request_json_mode
        self.timeout = timeout
        self.retry_policy = retry_policy or DEFAULT_RETRY_POLICY
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

        # Supply the HTTP client explicitly rather than letting the SDK build
        # its own. On Python 3.14 the SDK's default construction recurses until
        # it blows the stack — thousands of frames deep, surfacing as a bare
        # APIConnectionError("Connection error") indistinguishable from a
        # firewall block, while an identical request through an httpx client we
        # construct returns 200. Measured, not guessed: scripts/probe_sdk.py
        # tries both and reports which works.
        #
        # This is also strictly more predictable. The SDK's default client
        # picks up environment proxies and transport settings implicitly; ours
        # is visible in this file.
        # httpx2 first: that is what openai 3.x is built against. Falls back to
        # httpx for older SDKs, and to the SDK's own client if neither is
        # importable — never crashing merely because we could not supply one.
        http = http_library(("httpx2", "httpx"))
        if http is not None:
            kwargs["http_client"] = http.Client(timeout=self.timeout)

        self._client = OpenAI(**kwargs)
        return self._client

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def availability_error(self) -> Optional[str]:
        try:
            import openai  # noqa: F401
        except ImportError:
            return (
                "the 'openai' package is not installed in this environment — "
                "run:  pip install openai"
            )
        if not self._api_key:
            return (
                "OPENAI_API_KEY is empty. Put it in .env at the repository root "
                "(NOT .env.example, which is the committed template)."
            )
        return None

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

        def _send():
            return call_with_retry(
                lambda: client.chat.completions.create(**kwargs),
                self.retry_policy,
                on_wait=self._announce_wait,
            )

        t0 = time.perf_counter()
        try:
            resp = _send()
        except Exception as e:
            # One retry without json mode: some models reject response_format.
            # Any other failure is surfaced, not swallowed.
            if self.request_json_mode and "response_format" in str(e):
                kwargs.pop("response_format", None)
                try:
                    resp = _send()
                except Exception as e2:
                    raise LLMUnavailableError(explain_api_error(e2)) from e2
            else:
                raise LLMUnavailableError(explain_api_error(e)) from e
        latency_ms = (time.perf_counter() - t0) * 1000

        usage = getattr(resp, "usage", None)
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=self.model,
            provider=self.provider,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=latency_ms,
            # "length" means the reply was cut off by max_tokens. Downstream
            # this is the difference between "the model produced bad JSON" and
            # "the model produced good JSON that we truncated" — same symptom,
            # opposite fix.
            raw={"finish_reason": getattr(choice, "finish_reason", None)},
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
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.model = model or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-5"
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_policy = retry_policy or DEFAULT_RETRY_POLICY
        self._client = None

    def availability_error(self) -> Optional[str]:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return (
                "the 'anthropic' package is not installed in this environment — "
                "run:  pip install anthropic"
            )
        if not self._api_key:
            return (
                "ANTHROPIC_API_KEY is empty. Put it in .env at the repository root "
                "(NOT .env.example, which is the committed template)."
            )
        return None

    def complete(self, messages: Sequence[dict]) -> LLMResponse:
        import time

        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover - environment dependent
            raise LLMUnavailableError("The 'anthropic' package is not installed.") from e
        if not self._api_key:
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set.")
        if self._client is None:
            # Explicit HTTP client, for the same reason as OpenAIClient: the
            # SDK's default construction recurses on Python 3.14.
            kwargs = {"api_key": self._api_key, "timeout": self.timeout}
            # httpx first here: the anthropic SDK is built against it.
            http = http_library(("httpx", "httpx2"))
            if http is not None:
                kwargs["http_client"] = http.Client(timeout=self.timeout)
            self._client = anthropic.Anthropic(**kwargs)

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]

        t0 = time.perf_counter()
        try:
            resp = call_with_retry(
                lambda: self._client.messages.create(
                    model=self.model,
                    system=system,
                    messages=turns,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ),
                self.retry_policy,
                on_wait=self._announce_wait,
            )
        except Exception as e:
            raise LLMUnavailableError(explain_api_error(e)) from e
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


def http_library(prefer: Sequence[str] = ("httpx2", "httpx")):
    """The HTTP library available, in order of preference. None if neither is.

    openai 3.x depends on `httpx2`; openai 1.x and 2.x depended on `httpx`.
    Neither is declared in our requirements — both arrive through the SDK — so
    which one exists depends on which SDK version was installed, and a fresh
    install today has httpx2 while an older environment has httpx.

    Code that needs an HTTP client must therefore ask rather than assume.
    Importing `httpx` unconditionally is what broke a new contributor's setup:
    every check passed until the first line that said `import httpx`.

    The preference order matters when passing a client into an SDK — give it
    one built from the library it was itself compiled against, or the two
    generations' APIs meet somewhere unpleasant.
    """
    for name in prefer:
        try:
            return __import__(name)
        except ImportError:
            continue
    return None


class UnavailableClient(LLMClient):
    """A provider that could not be configured, represented rather than raised.

    Exists for one caller: the HTTP service, which must be able to start
    without an API key. Retrieval, the evidence viewer and the coverage
    endpoints touch no model at all, and refusing to boot the whole service
    because generation is unconfigured makes those unreachable too — which is
    both unhelpful and contrary to what the README promises.

    It is NOT a degraded provider and never silently substitutes anything:
    `complete()` raises immediately with the original reason, so an evaluation
    run that reached one would record LLM_ERROR and be marked invalid rather
    than producing output. `is_real` stays True because it is not a test
    double — the guard that keeps ScriptedClient out of reported results must
    not be weakened to accommodate this.
    """

    provider = "unavailable"

    def __init__(self, reason: str, provider_name: str = "unknown", model: str = "unconfigured"):
        self._reason = reason
        self.provider = provider_name
        self.model = model

    def availability_error(self) -> Optional[str]:
        return self._reason

    def complete(self, messages: Sequence[dict]) -> LLMResponse:
        raise LLMUnavailableError(self._reason)


def client_from_env(
    env_path: str | Path = DEFAULT_ENV_PATH,
    strict: bool = True,
) -> LLMClient:
    """Build the configured client. `LLM_PROVIDER` selects; default openai.

    Loads .env first, so every caller — CLI, smoke test, evaluation runner —
    picks up the same configuration without each having to remember to do it.

    With `strict=True` (the default) a misconfiguration raises, so it fails at
    startup rather than halfway through a 44-question evaluation run. The
    message names the one thing that is actually wrong.

    With `strict=False` the same problem is returned as an `UnavailableClient`
    carrying that message. Only the HTTP service uses this: it must start
    without a key so that retrieval and the evidence endpoints stay usable,
    and the failure then surfaces per-request as a 503 naming the cause.
    Nothing is silently substituted — the client refuses to generate.
    """
    env_found = load_dotenv(env_path)

    name = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if name not in PROVIDERS:
        message = f"Unknown LLM_PROVIDER '{name}'. Known providers: {sorted(PROVIDERS)}"
        if strict:
            raise LLMUnavailableError(message)
        return UnavailableClient(message, provider_name=name)

    client = PROVIDERS[name]()
    reason = client.availability_error()
    if reason is not None:
        hint = "" if env_found else f"\n  (no .env file found at {Path(env_path)})"
        message = f"provider '{name}' — {reason}{hint}"
        if strict:
            raise LLMUnavailableError(message)
        return UnavailableClient(message, provider_name=name, model=getattr(client, "model", "unconfigured"))
    return client
