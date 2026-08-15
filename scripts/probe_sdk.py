"""Pinpoint why the OpenAI SDK fails when raw httpx to the same endpoint works.

    python scripts/probe_sdk.py

`diagnose_network.py` established that the network, TLS, the key and the API
are all fine — an authenticated raw httpx request returns 200 — while the SDK
raises `APIConnectionError`. That narrows the fault to the SDK's own HTTP
stack, and this script finds out where.

It prints the versions of every library in that stack, then tries four
increasingly independent ways of making the same call:

    A  OpenAI()                       the SDK's default client
    B  OpenAI(http_client=...)        the SDK with an httpx client we built
    C  raw httpx GET  /v1/models      no SDK at all
    D  raw httpx POST /v1/chat/...    no SDK, and an actual generation

Whichever is the first to succeed tells you what to do:

    A works        nothing to fix
    B works        the SDK's default client construction is the problem;
                   pass an explicit http_client
    C or D only    the SDK is unusable in this environment and the project
                   should talk to the REST API directly

On a RecursionError the traceback is thousands of frames, so the repeating
cycle is extracted and printed instead — that names the module that loops.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.generation.llm import load_dotenv

HOST = "https://api.openai.com"
OK = "  OK   "
BAD = "  FAIL "


def show_versions() -> None:
    print("[versions]")
    print(f"  python      {sys.version.split()[0]}  ({sys.platform})")
    for mod in ("openai", "httpx", "httpcore", "h11", "anyio", "certifi", "pydantic"):
        try:
            m = __import__(mod)
            print(f"  {mod:<11} {getattr(m, '__version__', 'unknown')}")
        except ImportError:
            print(f"  {mod:<11} not installed")


def describe_failure(e: BaseException) -> None:
    """Print the useful part of an exception, including the repeating cycle of
    a RecursionError rather than ten thousand identical frames."""
    print(f"{BAD}{type(e).__name__}: {str(e)[:160]}")

    cause, depth = e.__cause__ or e.__context__, 0
    while cause is not None and depth < 4:
        print(f"       caused by: {type(cause).__name__}: {str(cause)[:160]}")
        root = cause
        cause = cause.__cause__ or cause.__context__
        depth += 1
    else:
        root = e

    tb = traceback.extract_tb(root.__traceback__)
    if not tb:
        return
    if isinstance(root, RecursionError) or len(tb) > 60:
        # Find the repeating cycle: the tail of a recursion traceback is the
        # same handful of frames over and over.
        tail = tb[-40:]
        seen, cycle = set(), []
        for f in tail:
            key = (f.filename, f.name)
            if key in seen:
                break
            seen.add(key)
            cycle.append(f)
        print(f"       recursion cycle ({len(tb)} frames total), repeating:")
        for f in cycle[:8]:
            print(f"         {Path(f.filename).name}:{f.lineno} in {f.name}()")
    else:
        for f in tb[-4:]:
            print(f"         {Path(f.filename).name}:{f.lineno} in {f.name}()")


def main() -> int:
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print("OPENAI_API_KEY is empty; set it in .env")
        return 1

    print("=" * 70)
    print("OPENAI SDK PROBE")
    print("=" * 70)
    show_versions()

    results = {}

    # --- A: the SDK's default client -------------------------------------
    print("\n[A] OpenAI() — the SDK's default client")
    try:
        from openai import OpenAI

        n = len(OpenAI(api_key=key, timeout=20).models.list().data)
        print(f"{OK}{n} models")
        results["A"] = True
    except Exception as e:
        describe_failure(e)
        results["A"] = False

    # --- B: the SDK with an httpx client we construct ---------------------
    print("\n[B] OpenAI(http_client=httpx.Client()) — our client, SDK's logic")
    try:
        import httpx
        from openai import OpenAI

        with httpx.Client(timeout=20) as hc:
            n = len(OpenAI(api_key=key, http_client=hc).models.list().data)
        print(f"{OK}{n} models")
        results["B"] = True
    except Exception as e:
        describe_failure(e)
        results["B"] = False

    # --- C: raw httpx, no SDK --------------------------------------------
    print("\n[C] raw httpx GET /v1/models — no SDK")
    try:
        import httpx

        r = httpx.get(f"{HOST}/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
        print(f"{OK}HTTP {r.status_code}, {len(r.json().get('data', []))} models")
        results["C"] = r.status_code == 200
    except Exception as e:
        describe_failure(e)
        results["C"] = False

    # --- D: raw httpx, an actual generation -------------------------------
    print("\n[D] raw httpx POST /v1/chat/completions — a real generation, no SDK")
    try:
        import httpx

        r = httpx.post(
            f"{HOST}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                "max_tokens": 5,
                "temperature": 0,
            },
            timeout=30,
        )
        if r.status_code == 200:
            reply = r.json()["choices"][0]["message"]["content"].strip()
            print(f"{OK}model replied: {reply!r}")
            results["D"] = True
        else:
            print(f"{BAD}HTTP {r.status_code}: {r.text[:200]}")
            results["D"] = False
    except Exception as e:
        describe_failure(e)
        results["D"] = False

    # --- verdict ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if results.get("A"):
        print("  The SDK works. Nothing to change.")
    elif results.get("B"):
        print("  The SDK works when given an explicit httpx client, and fails")
        print("  when it builds its own. Fix: pass http_client= in OpenAIClient.")
    elif results.get("D"):
        print("  The SDK is unusable here, but the REST API works perfectly and")
        print("  can generate. Fix: talk to the API directly over httpx and drop")
        print("  the SDK dependency for this project.")
        print("\n  That is a supported path — the endpoint is a documented, stable")
        print("  HTTP interface. Say the word and I will implement it.")
    elif results.get("C"):
        print("  Reads work but generation does not — check the model name in")
        print("  .env (OPENAI_MODEL) and that your account can use it.")
    else:
        print("  Nothing reached the API. Re-run scripts/diagnose_network.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
