"""List the models the configured provider will actually serve you.

    python scripts/list_models.py

Exists because of a specific, wasteful failure mode: a provider returns
HTTP 404 for a model name that is documented, current and spelled correctly —
because that particular key, tier or region does not have it. The error says
"model not found", which reads like a typo, so the natural response is to check
the spelling again rather than to ask what is actually available.

Reading the provider's docs does not settle it either. Documentation lists what
the provider offers; this lists what your key is entitled to, which is a
smaller set and the one that matters.

Uses whatever OPENAI_BASE_URL points at, so it works for OpenAI, Azure, Gemini,
Groq and OpenRouter alike. Listing models is free everywhere, so this never
costs anything — and for the same reason it proves nothing about quota. Only a
real generation does that: scripts/diagnose_network.py, step 10.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.generation.llm import http_library, load_dotenv

DEFAULT_BASE = "https://api.openai.com/v1"


def main() -> int:
    load_dotenv()

    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print("OPENAI_API_KEY is empty. Put it in .env and re-run.")
        return 1

    base = (os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE).rstrip("/")
    configured = os.getenv("OPENAI_MODEL", "")

    httpx = http_library()
    if httpx is None:
        print("Neither httpx nor httpx2 is installed — run: pip install -r requirements.txt")
        return 1

    print(f"endpoint : {base}")
    print(f"OPENAI_MODEL in .env : {configured or '(unset)'}\n")

    try:
        r = httpx.get(
            f"{base}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001 - reported, not raised
        print(f"request failed: {type(e).__name__}: {e}")
        print("Run scripts/diagnose_network.py to find which layer is broken.")
        return 1

    if r.status_code == 404:
        print("This endpoint does not implement /models, so the list cannot be")
        print("fetched. Check the provider's own console for model names.")
        return 1
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:300]}")
        return 1

    ids = sorted(m.get("id", "") for m in r.json().get("data", []))
    if not ids:
        print("The endpoint returned an empty list.")
        return 1

    print(f"{len(ids)} model(s) available to this key:\n")
    for mid in ids:
        marker = "  <-- OPENAI_MODEL" if mid == configured or mid.endswith(f"/{configured}") else ""
        print(f"  {mid}{marker}")

    # The whole point: say plainly whether the configured name is in the list.
    if configured and not any(m == configured or m.endswith(f"/{configured}") for m in ids):
        print(f"\n'{configured}' is NOT in that list — which is why generation returns 404.")
        print("Copy one of the names above into OPENAI_MODEL in .env, exactly as shown.")
        return 1
    if configured:
        print(f"\n'{configured}' is available. If generation still fails, the cause is")
        print("quota or permissions, not the model name — diagnose_network.py step 10.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
