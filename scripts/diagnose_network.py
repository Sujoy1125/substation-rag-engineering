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
    8. auth      does an authenticated request work via raw httpx...
    9. SDK       ...and via the OpenAI SDK?

Steps 8 and 9 are separate on purpose. If raw httpx succeeds with the same key
and the SDK does not, then the key, the route and TLS are all fine and the
fault is in the SDK's own HTTP stack — a completely different fix from
anything network-related. `APIConnectionError` is a wrapper; step 9 unwraps
the cause chain, because the wrapper's name is never the actual failure.

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

from src.generation.llm import load_dotenv

HOST = "api.openai.com"
PORT = 443

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
        print(f"{WARN}OPENAI_BASE_URL = {base!r}")
        print("       Requests go there, NOT to api.openai.com. Remove it unless")
        print("       you are deliberately using a proxy or local model.")

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
    try:
        import httpx

        r = httpx.get(f"https://{HOST}/v1/models", timeout=15)
        if r.status_code == 401:
            print(f"{OK}reached the API (401 Unauthorized, as expected without a key)")
        elif r.status_code == 403:
            print(f"{WARN}403 Forbidden — reached something, but access is blocked.")
            print("       Often an egress proxy rather than OpenAI itself.")
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
    try:
        import httpx

        r = httpx.get(
            f"https://{HOST}/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        )
        if r.status_code == 200:
            print(f"{OK}200 — the key works and the network allows it")
            raw_ok = True
        elif r.status_code == 401:
            print(f"{BAD}401 Unauthorized — the key is rejected by OpenAI")
            print("       Revoked, mistyped, or from a different account.")
            print("       Generate a new one: https://platform.openai.com/api-keys")
            return 1
        elif r.status_code == 429:
            print(f"{BAD}429 — the key is valid but out of quota/credit")
            print("       https://platform.openai.com/settings/organization/billing")
            return 1
        else:
            print(f"{WARN}status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"{BAD}{type(e).__name__}: {str(e)[:200]}")
        print("       Even a raw authenticated request fails, so this is the")
        print("       network refusing authenticated traffic, not the SDK.")
        return 1

    # --- the SDK itself ------------------------------------------------
    step("Authenticated request via the OpenAI SDK")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, timeout=20)
        models = client.models.list()
        names = [m.id for m in models.data][:3]
        print(f"{OK}SDK works — {len(models.data)} models, e.g. {', '.join(names)}")
    except Exception as e:
        print(f"{BAD}{type(e).__name__}: {str(e)[:200]}")
        # openai wraps the real failure; the wrapper name is never the cause.
        cause, depth = e.__cause__ or e.__context__, 0
        while cause is not None and depth < 6:
            print(f"       caused by: {type(cause).__name__}: {str(cause)[:200]}")
            cause = cause.__cause__ or cause.__context__
            depth += 1
        if raw_ok:
            print()
            print("       Raw httpx with the same key succeeded, so the key, the")
            print("       network and TLS are all fine — the SDK's own HTTP client")
            print("       is what fails. Usually one of:")
            print("         - an old/conflicting httpx: pip install -U openai httpx")
            print("         - a stale openai version:   pip show openai")
            print("         - antivirus or endpoint security filtering the process")
        return 1

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED — the live evaluation should run.")
    print("  python experiments/run_generation_eval.py --live --limit 3")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
