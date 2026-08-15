"""Verify this machine can run the project.

    python scripts/check_setup.py

Written for a team where not everyone has the same Python, the same OS, or the
same tolerance for a stack trace. It answers one question — "is my setup
correct?" — in a form that names the fix rather than the symptom.

It checks, in dependency order, and never stops at the first failure: knowing
that three things are wrong is more useful than being told about the first one
three times.

Nothing here makes a paid API call. The key check confirms a key is present
and well-formed; whether the account has credit is a question only a real
generation can answer, and that lives in scripts/diagnose_network.py.
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OK = "  OK   "
BAD = "  FAIL "
WARN = "  WARN "

MIN_PYTHON = (3, 10)

CORE = [
    ("openpyxl", "reads the frozen knowledge base workbooks"),
    ("numpy", "vector maths in retrieval"),
    ("rank_bm25", "the BM25 retriever"),
    ("sklearn", "TF-IDF baseline and calibration"),
    ("openai", "the LLM client (also used for Azure/Gemini/Groq/OpenRouter)"),
    ("fastapi", "the HTTP service"),
    ("uvicorn", "runs the HTTP service"),
    ("pytest", "the test suite"),
]

failures: list[str] = []
warnings: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


def check_python() -> None:
    print("[1] Python version")
    v = sys.version_info
    print(f"       {platform.python_version()} on {platform.system()}")
    if (v.major, v.minor) < MIN_PYTHON:
        print(f"{BAD}too old — this project needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer")
        fail(f"Install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ and recreate the virtual environment.")
    else:
        print(f"{OK}supported")


def check_venv() -> None:
    print("\n[2] Virtual environment")
    active = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if active:
        print(f"{OK}active — {sys.prefix}")
    else:
        print(f"{WARN}not active; packages would install system-wide")
        warn(
            "Create and activate one first:\n"
            "         python -m venv venv\n"
            "         venv\\Scripts\\activate      (Windows)\n"
            "         source venv/bin/activate   (macOS / Linux)"
        )


def check_packages() -> None:
    print("\n[3] Core dependencies")
    missing = []
    for mod, why in CORE:
        try:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "")
            print(f"{OK}{mod:<12} {version:<10} {why}")
        except ImportError:
            print(f"{BAD}{mod:<12} {'':<10} {why}")
            missing.append(mod)
    if missing:
        fail("Install the dependencies:  pip install -r requirements.txt")


def check_optional() -> None:
    print("\n[4] Optional dependencies (dense retrieval — not required)")
    try:
        importlib.import_module("sentence_transformers")
        print(f"{OK}sentence-transformers present — dense retrieval code can run")
    except ImportError:
        print(f"{OK}absent, and that is fine")
        print("       Dense retrieval is UNMEASURED and not needed to run anything.")
        print("       Only if you are working on it:  pip install -r requirements-dense.txt")


def check_knowledge_base() -> None:
    print("\n[5] Knowledge base")
    kb = REPO_ROOT / "KB_v1.1" / "KB_v1.1_final" / "knowledge_chunks.xlsx"
    if not kb.exists():
        print(f"{BAD}not found at {kb.relative_to(REPO_ROOT)}")
        fail(
            "The knowledge base workbook is missing. If you unzipped this project, the\n"
            "         archive was incomplete — clone the repository instead:\n"
            "         git clone https://github.com/Sujoy1125/substation-rag-engineering"
        )
        return
    try:
        from src.ingestion.kb_loader import load_chunks

        chunks, report = load_chunks(str(kb))
        print(f"{OK}loaded {len(chunks)} chunks from {kb.name}")
        if len(chunks) != 1745:
            print(f"{WARN}expected 1745 — this workbook may not be KB_v1.1")
            warn("Chunk count differs from the frozen KB_v1.1. Check which version you have.")
        if not report.ok():
            print(f"{WARN}{len(report.malformed_rows)} malformed row(s)")
    except Exception as e:  # noqa: BLE001 - reported, not raised
        print(f"{BAD}could not load: {type(e).__name__}: {e}")
        fail("The knowledge base could not be read. Re-clone rather than repairing by hand.")


def check_retrieval() -> None:
    """The end-to-end check that needs no API key at all."""
    print("\n[6] Retrieval (no API key needed)")
    try:
        from src.generation.pipeline import build_default_pipeline

        pipe = build_default_pipeline(llm=None)
        results = pipe.retrieve("transformer insulating oil BDV test frequency", top_k=3)
        if not results:
            print(f"{BAD}retrieved nothing")
            fail("Retrieval returned no results — the index did not build.")
            return
        print(f"{OK}retrieved {len(results)} chunks; top hit {results[0].chunk.chunk_id} "
              f"(score {results[0].score:.2f})")
    except Exception as e:  # noqa: BLE001
        print(f"{BAD}{type(e).__name__}: {e}")
        fail("Retrieval failed to run. Check steps 3 and 5 above first.")


def check_env() -> None:
    print("\n[7] API key (.env)")
    from src.generation.llm import load_dotenv

    env_path = REPO_ROOT / ".env"
    example = REPO_ROOT / ".env.example"
    if not env_path.exists():
        print(f"{WARN}no .env file")
        print("       Everything except answer generation still works without one.")
        warn(
            "To enable generation, copy the template and add your own key:\n"
            "         copy .env.example .env      (Windows)\n"
            "         cp .env.example .env        (macOS / Linux)\n"
            "         Put the key in .env — NEVER in .env.example, which is committed."
        )
        return
    load_dotenv(env_path)
    import os

    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print(f"{WARN}.env exists but OPENAI_API_KEY is empty")
        warn("Add your key to .env. Generation will not run without it.")
        return
    print(f"{OK}key present ({len(key)} chars, starts {key[:7]}…)")
    base = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    print(f"       model {model}" + (f" via {base}" if base else " via the default OpenAI endpoint"))
    print("       Whether the account has credit is a different question — a key can pass")
    print("       every check and still fail on the first generation. To find out:")
    print("           python scripts/diagnose_network.py")

    # The mistake that cost this project a revoked key.
    if example.exists() and "sk-" in example.read_text(encoding="utf-8", errors="ignore"):
        print(f"{BAD}a key appears to be in .env.example, which IS committed to git")
        fail(
            "Remove the key from .env.example and REVOKE it — anything committed to git\n"
            "         is compromised permanently, even after the file is changed."
        )


def main() -> int:
    print("=" * 70)
    print("SETUP CHECK — Substation O&M Assistant")
    print("=" * 70)

    check_python()
    check_venv()
    check_packages()
    check_optional()
    check_knowledge_base()
    check_retrieval()
    check_env()

    print("\n" + "=" * 70)
    if failures:
        print(f"NOT READY — {len(failures)} problem(s) to fix")
        print("=" * 70)
        for i, f in enumerate(failures, 1):
            print(f"  {i}. {f}")
        return 1

    print("READY")
    print("=" * 70)
    print("  Run the tests:      python -m pytest -q")
    print("  Start the service:  uvicorn src.api.service:build_default_app --factory --reload")
    print("  Then open:          http://127.0.0.1:8000/")
    if warnings:
        print(f"\n  {len(warnings)} note(s):")
        for i, w in enumerate(warnings, 1):
            print(f"    {i}. {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
