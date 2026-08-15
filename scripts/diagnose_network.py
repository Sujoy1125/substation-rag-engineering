"""Find out exactly which layer is breaking between you and the LLM API.

    python scripts/diagnose_network.py

"Connection error" from the OpenAI client is a single message covering at
least six distinct failures, and the fix is different for each. This walks the
layers in order and stops being useful only where it actually breaks:

    1. .env loaded, key present
    2. proxy environment variables
    3. DNS      does api.openai.com resolve?
    4. TCP      can we open port 443?
    5. TLS      does the handshake succeed, and WHO signed the certificate?
    6. HTTP     does an unauthenticated request reach the API?
    7. API      does the key work?

Step 5 is the one that catches the failure most people miss. Campus and
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


def step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")


def main() -> int:
    print("=" * 70)
    print("LLM CONNECTIVITY DIAGNOSTIC")
    print("=" * 70)

    # --- 1. configuration ------------------------------------------------
    step(1, "Configuration (.env)")
    found = load_dotenv()
    print(f"{OK if found else WARN}.env file: {'loaded' if found else 'NOT FOUND at ' + str(REPO_ROOT / '.env')}")

    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print(f"{BAD}OPENAI_API_KEY is empty")
        print("       Put the key in .env (NOT .env.example) and re-run.")
        return 1
    print(f"{OK}OPENAI_API_KEY present ({len(key)} chars, starts {key[:7]}…)")

    base = os.getenv("OPENAI_BASE_URL", "")
    if base:
        print(f"{WARN}OPENAI_BASE_URL is set to {base!r}")
        print("       Requests go there, NOT to api.openai.com. Unset it unless")
        print("       you are deliberately using a proxy or local model.")

    # --- 2. proxies ------------------------------------------------------
    step(2, "Proxy environment")
    proxies = {k: v for k, v in os.environ.items() if k.lower() in
               ("http_proxy", "https_proxy", "all_proxy", "no_proxy")}
    if proxies:
        for k, v in proxies.items():
            print(f"{WARN}{k} = {v}")
        print("       A proxy is configured. If it requires authentication or")
        print("       blocks this host, that is your failure.")
    else:
        print(f"{OK}no proxy variables set (direct connection expected)")

    # --- 3. DNS ----------------------------------------------------------
    step(3, f"DNS resolution of {HOST}")
    try:
        addrs = sorted({ai[4][0] for ai in socket.getaddrinfo(HOST, PORT, proto=socket.IPPROTO_TCP)})
        print(f"{OK}resolves to {', '.join(addrs[:4])}")
    except socket.gaierror as e:
        print(f"{BAD}cannot resolve: {e}")
        print("       No DNS. Check you are online, or that a captive portal")
        print("       (hotel/campus login page) is not intercepting requests.")
        return 1

    # --- 4. TCP ----------------------------------------------------------
    step(4, f"TCP connection to {HOST}:{PORT}")
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

    # --- 5. TLS + certificate issuer -------------------------------------
    step(5, "TLS handshake and certificate issuer")
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

    # --- 6. HTTP reachability (no auth) ----------------------------------
    step(6, "HTTP request to the API (no key — 401 is the success case)")
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

    # --- 7. the key ------------------------------------------------------
    step(7, "Authenticated request with your key")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, timeout=20)
        models = client.models.list()
        names = [m.id for m in models.data][:3]
        print(f"{OK}key works — {len(models.data)} models available, e.g. {', '.join(names)}")
    except Exception as e:
        msg = str(e)
        print(f"{BAD}{type(e).__name__}: {msg[:200]}")
        if "401" in msg or "invalid_api_key" in msg:
            print("       The key is wrong or revoked. Generate a new one at")
            print("       https://platform.openai.com/api-keys")
        elif "429" in msg or "quota" in msg.lower():
            print("       The key is valid but has no credit. Add billing at")
            print("       https://platform.openai.com/settings/organization/billing")
        return 1

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED — the live evaluation should run.")
    print("  python experiments/run_generation_eval.py --live --limit 3")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
