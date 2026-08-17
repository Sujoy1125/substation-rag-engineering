"""Find out exactly which layer is breaking between you and the LLM API.

    python scripts/diagnose_network.py

"Connection error" from the OpenAI client is a single message covering at
least six distinct failures, and the fix is different for each. This walks the
layers in order and stops being useful only where it actually breaks:

    1. config    .env loaded, key present
    2. key       any invisible characters that an HTTP header cannot carry?
    3. proxy     proxy environment variables
    4. DNS       does api.openai.com resolve?
    5. TCP       can we open port 443?
    6. TLS       does the handshake succeed, and WHO signed the certificate?
    7. HTTP      does an UNauthenticated request reach the API?
    8. auth      does an authenticated request work via raw httpx?
    9. client    does the project's own client build?
   10. generate  does a real generation succeed?

Steps 8 and 9 are separate on purpose. If raw httpx succeeds with the same key
and the project's client does not, the key, route and TLS are all fine and the
fault is in the client stack — a completely different fix from anything
network-related. `APIConnectionError` is a wrapper; the cause chain is
unwrapped, because the wrapper's name is never the actual failure.

Step 10 exists because an earlier version of this script passed every check
against a key with no credit. Listing models is free; generating is not. Only
a real generation proves the system can actually run, so the diagnostic makes
one — and step 9 uses `client_from_env()`, the same call the evaluation runner
makes, so the diagnostic cannot pass on a path the application never uses.

The TLS step is the one that catches the failure most people miss. Campus and
corporate networks frequently intercept HTTPS with their own certificate
authority. Browsers accept it because the CA is installed in the OS trust
store; Python does not, because it ships its own via certifi. The symptom is
exactly this — the site works in Chrome, and every Python client reports a
bare "Connection error". This script names the interceptor.

Nothing here sends your key anywhere except api.openai.com, and the key is
never printed.
"""
from __future__ import annotations

import os
import socket
import ssl
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.generation.llm import http_library, load_dotenv

DEFAULT_HOST = "api.openai.com"
PORT = 443

# Filled in by step 1 from OPENAI_BASE_URL, so every later step tests the
# server this installation actually talks to. Diagnosing api.openai.com while
# the project is configured for Gemini, Azure or OpenRouter reports failures
# about a host that is never contacted — which is worse than no diagnostic,
# because it sends the reader somewhere real to look.
HOST = DEFAULT_HOST

# The FULL base URL, path included. Steps 7-8 must use this rather than
# rebuilding "https://{HOST}/v1" — Gemini's OpenAI-compatible surface lives at
# /v1beta/openai/, and /v1 on the same host is Google's native API, which
# rejects a Bearer token. Dropping the path turns a working key into a 401.
BASE_URL = f"https://{DEFAULT_HOST}/v1"


def _host_from_base_url(base: str) -> str:
    """Extract the hostname from a base URL, defaulting if it is unparseable."""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(base)
        return parsed.hostname or DEFAULT_HOST
    except Exception:  # noqa: BLE001 - a bad URL is reported by step 1
        return DEFAULT_HOST

OK = "  OK   "
BAD = "  FAIL "
WARN = "  WARN "


_step_no = [0]


def step(title: str) -> None:
    """Auto-numbered so steps can be reordered without the numbers drifting."""
    _step_no[0] += 1
    print(f"\n[{_step_no[0]}] {title}")


def main() -> int:
    print("=" * 70)
    print("LLM CONNECTIVITY DIAGNOSTIC")
    print("=" * 70)

    # --- configuration ------------------------------------------------
    step("Configuration (.env)")
    found = load_dotenv()
    print(f"{OK if found else WARN}.env file: {'loaded' if found else 'NOT FOUND at ' + str(REPO_ROOT / '.env')}")

    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print(f"{BAD}OPENAI_API_KEY is empty")
        print("       Put the key in .env (NOT .env.example) and re-run.")
        return 1
    print(f"{OK}OPENAI_API_KEY present ({len(key)} chars, starts {key[:7]}…)")

    # An empty-but-present OPENAI_BASE_URL is the trap: the SDK reads this
    # variable itself and treats "" as a base URL, producing a request to a
    # URL with no scheme and an APIConnectionError that reads like a firewall.
    if "OPENAI_BASE_URL" in os.environ:
        base = os.environ["OPENAI_BASE_URL"]
        if not base.strip():
            print(f"{BAD}OPENAI_BASE_URL is set but EMPTY")
            print("       The OpenAI SDK reads this variable directly and treats an")
            print("       empty string as a base URL, building a request to a URL")
            print("       with no scheme. It surfaces as 'Connection error', which")
            print("       looks like a network fault and is not one.")
            print("       Fix: delete or comment out the OPENAI_BASE_URL line in .env")
            return 1
        if not base.startswith(("http://", "https://")):
            print(f"{BAD}OPENAI_BASE_URL = {base!r} has no http:// or https:// scheme")
            return 1
        # A configured base URL is normal — the client speaks plain OpenAI
        # protocol, so Azure, Gemini, Groq and OpenRouter all work through it.
        # Recognised endpoints are reported as information; anything else gets
        # a warning, since a typo here is otherwise invisible.
        global HOST, BASE_URL
        HOST = _host_from_base_url(base)
        BASE_URL = base.rstrip("/")
        KNOWN = {
            "generativelanguage.googleapis.com": "Google Gemini",
            "openrouter.ai": "OpenRouter",
            "api.groq.com": "Groq",
        }
        provider = KNOWN.get(HOST) or ("Azure OpenAI" if HOST.endswith(".openai.azure.com") else None)
        if provider:
            print(f"{OK}OPENAI_BASE_URL points at {provider} ({HOST})")
            print(f"       All checks below test {HOST}, not api.openai.com.")
        else:
            print(f"{WARN}OPENAI_BASE_URL = {base!r}")
            print(f"       Unrecognised endpoint. Checks below will test {HOST}.")
            print("       If that is a typo, requests will fail in ways that look")
            print("       like a network fault. Remove the line to use OpenAI.")

    # --- key hygiene --------------------------------------------------
    # A key that survived copy-paste through a browser, a chat window and a
    # text editor can carry characters that are invisible on screen but fatal
    # in an HTTP header. Checked before blaming the network.
    step("API key hygiene")
    problems = []
    if key != key.strip():
        problems.append("leading/trailing whitespace")
    non_ascii = [(i, ch) for i, ch in enumerate(key) if ord(ch) > 126 or ord(ch) < 32]
    if non_ascii:
        problems.append(
            "non-ASCII or control characters at positions "
            + ", ".join(f"{i} (U+{ord(c):04X})" for i, c in non_ascii[:5])
        )
    if any(q in key for q in ('"', "'", "“", "”")):
        problems.append("quote characters inside the key")
    if problems:
        for p in problems:
            print(f"{BAD}{p}")
        print("       An HTTP header cannot carry these. Re-copy the key from")
        print("       https://platform.openai.com/api-keys straight into .env,")
        print("       with no quotes and nothing after it on the line.")
        return 1
    print(f"{OK}clean: ASCII only, no stray whitespace or quotes")

    # --- proxies ------------------------------------------------------
    step("Proxy environment")
    proxies = {k: v for k, v in os.environ.items() if k.lower() in
               ("http_proxy", "https_proxy", "all_proxy", "no_proxy")}
    if proxies:
        for k, v in proxies.items():
            print(f"{WARN}{k} = {v}")
        print("       A proxy is configured. If it requires authentication or")
        print("       blocks this host, that is your failure.")
    else:
        print(f"{OK}no proxy variables set (direct connection expected)")

    # --- DNS ----------------------------------------------------------
    step(f"DNS resolution of {HOST}")
    try:
        addrs = sorted({ai[4][0] for ai in socket.getaddrinfo(HOST, PORT, proto=socket.IPPROTO_TCP)})
        print(f"{OK}resolves to {', '.join(addrs[:4])}")
    except socket.gaierror as e:
        print(f"{BAD}cannot resolve: {e}")
        print("       No DNS. Check you are online, or that a captive portal")
        print("       (hotel/campus login page) is not intercepting requests.")
        return 1

    # --- TCP ----------------------------------------------------------
    step(f"TCP connection to {HOST}:{PORT}")
    try:
        with socket.create_connection((HOST, PORT), timeout=10):
            print(f"{OK}port {PORT} is open")
    except socket.timeout:
        print(f"{BAD}timed out")
        print("       A firewall is dropping the connection silently — the")
        print("       classic signature of a campus/corporate block. Try a")
        print("       phone hotspot to confirm.")
        return 1
    except OSError as e:
        print(f"{BAD}refused: {e}")
        print("       Something actively rejected the connection.")
        return 1

    # --- TLS + certificate issuer -------------------------------------
    step("TLS handshake and certificate issuer")
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((HOST, PORT), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=HOST) as tls:
                cert = tls.getpeercert()
        issuer = dict(x[0] for x in cert.get("issuer", ()))
        org = issuer.get("organizationName", "unknown")
        cn = issuer.get("commonName", "unknown")
        print(f"{OK}handshake succeeded")
        print(f"       certificate issued by: {org} / {cn}")
        looks_intercepted = not any(
            k in f"{org} {cn}".lower()
            for k in ("digicert", "let's encrypt", "google", "globalsign", "amazon", "sectigo", "isrg")
        )
        if looks_intercepted:
            print(f"{WARN}that issuer is not a public CA — this connection is")
            print("       probably being intercepted by a network appliance.")
    except ssl.SSLCertVerificationError as e:
        print(f"{BAD}certificate verification failed: {e}")
        print()
        # Two very different causes produce this identical message, and the
        # advice for one is useless for the other. Split them by platform,
        # because on macOS the overwhelmingly likely cause is a fresh
        # python.org install whose root certificates were never installed —
        # not network interception.
        if sys.platform == "darwin":
            import glob

            print("       MOST LIKELY: this Python has no root certificates installed.")
            print("       Python downloaded from python.org does not use the macOS")
            print("       keychain. It ships its own certificate bundle and a script")
            print("       to install it, which the .pkg installer does NOT run for you.")
            print()
            print("       Fix — run this once, then re-run this diagnostic:")
            found = sorted(glob.glob("/Applications/Python*/Install Certificates.command"))
            if found:
                for path in found:
                    print(f'           "{path}"')
            else:
                print('           "/Applications/Python 3.12/Install Certificates.command"')
                print("           (adjust the version to match yours)")
            print()
            print("       Or, equivalently, from inside your virtual environment:")
            print("           pip install --upgrade certifi")
            print()
            print("       LESS LIKELY: your network intercepts HTTPS with its own")
            print("       certificate authority — common on campus wifi. If the fix")
            print("       above does not help, try a phone hotspot to confirm.")
        else:
            print("       THIS IS THE COMMON CAMPUS-NETWORK FAILURE.")
            print("       Your network is intercepting HTTPS with its own certificate")
            print("       authority. Browsers accept it (the CA is in the Windows trust")
            print("       store); Python does not (it uses certifi's bundle).")
            print()
            print("       Fixes, best first:")
            print("         - use a phone hotspot or a network without interception")
            print("         - export the network's root CA and point Python at it:")
            print("             $env:SSL_CERT_FILE = 'C:\\path\\to\\corporate-ca.pem'")
            print("             $env:REQUESTS_CA_BUNDLE = 'C:\\path\\to\\corporate-ca.pem'")
            print("         - pip install pip-system-certs   (makes Python use the")
            print("           Windows trust store, which already has the CA)")
        return 1
    except Exception as e:
        print(f"{BAD}TLS failed: {type(e).__name__}: {e}")
        return 1

    # --- HTTP reachability (no auth) ----------------------------------
    step("HTTP request to the API (no key — 401 is the success case)")
    httpx = http_library()
    if httpx is None:
        print(f"{BAD}neither httpx nor httpx2 is installed")
        print("       Both arrive with the openai package; if neither is present the")
        print("       install did not finish.  Fix:  pip install -r requirements.txt")
        return 1
    try:

        r = httpx.get(f"{BASE_URL}/models", timeout=15)
        if r.status_code in (401, 403):
            # 403 is a valid "reached it, refused without credentials" from
            # several compatible endpoints, not only a proxy block.
            print(f"{OK}reached the endpoint ({r.status_code} without a key, as expected)")
        elif r.status_code == 404:
            print(f"{WARN}404 — this endpoint does not expose /models")
            print("       Not a fault: several OpenAI-compatible providers implement")
            print("       only the chat endpoint. Step 10 is the check that matters.")
        else:
            print(f"{WARN}unexpected status {r.status_code}")
    except Exception as e:
        print(f"{BAD}{type(e).__name__}: {str(e)[:160]}")
        print("       TLS worked but the HTTP request did not complete.")
        return 1

    # --- authenticated request, bypassing the SDK ---------------------
    # Separates "the network rejects authenticated traffic" from "the SDK
    # cannot make the call". Step 6 already proved unauthenticated requests
    # get through, so a failure here is about the request, not the route.
    step("Authenticated request via raw httpx (bypasses the OpenAI SDK)")
    raw_ok = False
    httpx = http_library()
    if httpx is None:
        print(f"{BAD}neither httpx nor httpx2 is installed")
        print("       Both arrive with the openai package; if neither is present the")
        print("       install did not finish.  Fix:  pip install -r requirements.txt")
        return 1
    try:

        r = httpx.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        )
        where = "your provider" if HOST != DEFAULT_HOST else "OpenAI"
        if r.status_code == 200:
            print(f"{OK}200 — the key works and the network allows it")
            raw_ok = True
        elif r.status_code == 404:
            # Not every compatible endpoint implements /models. Never abort on
            # this: step 10 makes a real generation, which is the actual test.
            print(f"{WARN}404 — this endpoint does not implement /models")
            print("       Nothing is proven either way. Continuing to step 10,")
            print("       which makes a real generation and settles it.")
        elif r.status_code == 401:
            print(f"{WARN}401 Unauthorized — {where} rejected this key here")
            print("       Revoked, mistyped, or issued for a different service.")
            if HOST != DEFAULT_HOST:
                print("       NOTE: some compatible endpoints authenticate the model")
                print("       listing differently from generation. Continuing to")
                print("       step 10 rather than concluding from this alone.")
            else:
                print("       Generate a new one: https://platform.openai.com/api-keys")
                return 1
        elif r.status_code == 429:
            print(f"{BAD}429 — the key is valid but out of quota/credit")
            print("       Add credit with your provider, then re-run.")
            return 1
        else:
            print(f"{WARN}status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"{BAD}{type(e).__name__}: {str(e)[:200]}")
        print("       Even a raw authenticated request fails, so this is the")
        print("       network refusing authenticated traffic, not the SDK.")
        return 1

    # --- the project's own client ----------------------------------------
    # Deliberately the REAL path — client_from_env(), the same call the
    # evaluation runner makes. A diagnostic that constructs its own client
    # tests a code path the application never uses, and can pass while the
    # application fails.
    step("The project's own client (src.generation.llm.client_from_env)")
    try:
        from src.generation.llm import client_from_env

        client = client_from_env()
        print(f"{OK}built: provider={client.provider} model={client.model}")
    except Exception as e:
        print(f"{BAD}{type(e).__name__}: {str(e)[:300]}")
        cause, depth = e.__cause__ or e.__context__, 0
        while cause is not None and depth < 6:
            print(f"       caused by: {type(cause).__name__}: {str(cause)[:200]}")
            cause = cause.__cause__ or cause.__context__
            depth += 1
        return 1

    # --- an actual generation --------------------------------------------
    # THE STEP THAT WAS MISSING. Every check above can pass on a key with no
    # credit, because listing models is free and generating is not. The first
    # real evidence of a quota problem is a generation, so the diagnostic has
    # to make one.
    step("A real generation (the only check that detects an empty quota)")
    try:
        resp = client.complete(
            [{"role": "user", "content": "Reply with the single word: ready"}]
        )
        text = (resp.text or "").strip()
        print(f"{OK}model replied {text[:40]!r}")
        if resp.completion_tokens is not None:
            print(f"       tokens: {resp.prompt_tokens} in / {resp.completion_tokens} out")
    except Exception as e:
        print(f"{BAD}{str(e)[:600]}")
        if raw_ok:
            print()
            print("       Note: reads succeeded earlier. Listing models is free and")
            print("       generating is not, so a key can pass every connectivity")
            print("       check and still fail here.")
        return 1

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED — including a real generation.")
    print("  python experiments/run_generation_eval.py --live --limit 3")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
