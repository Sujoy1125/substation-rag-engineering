# Engineering Blockers

## Resolved

*(none yet — both items below remain open as of Phase 4)*

## Active

### 1. Dense/embedding retrieval unreachable in this Claude sandbox

**Status:** Sandbox blocker — local execution path available.
This is not a project architecture failure; it's a network-egress
restriction specific to this sandbox.

Confirmed directly (Phase 4):
```
curl -sS -o /dev/null -w "%{http_code}" https://huggingface.co   -> 403
curl -sS -o /dev/null -w "%{http_code}" https://api.openai.com   -> 403
curl -sS -o /dev/null -w "%{http_code}" https://pypi.org         -> 200
```

**Re-confirmed 2026-08-15**, independently, in a different sandbox during the
generation phase — `sentence_transformers` imports fine (PyPI reachable) but
loading `BAAI/bge-small-en-v1.5` still fails with `ProxyError 403 Forbidden`.
So this is a stable property of the sandbox egress policy, not a transient
outage. **Dense and hybrid retrieval remain unmeasured, and nothing anywhere in
this repository claims otherwise.**

The unblock is the local Windows run below; it needs no code changes.
`pip install torch sentence-transformers` succeeds (pulls from PyPI/pythonhosted).
Loading `BAAI/bge-small-en-v1.5` then fails with a clean, caught
`ModelUnavailableError` (`src/embeddings/provider.py`) — no fake vectors
are produced, no metrics are fabricated.

**Code impact:** `DenseRetriever` and `HybridRRFRetriever` are fully
implemented and unit-tested against a mock provider (interface-only,
never used for Recall@K). They will run for real the moment model weights
are reachable — no retrieval code needs to change.

**Exact commands to run locally on Windows** (target deployment
environment) to unblock dense + hybrid:

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install sentence-transformers torch

# 3. Download the embedding model (requires normal internet access —
#    this step is what's blocked in the Claude sandbox, not on a normal
#    Windows machine)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# 4. Run the retrieval benchmark — this will now include dense + hybrid
#    results automatically (src/retrieval/benchmark.py already attempts
#    them and only reports BLOCKED if weights are still unreachable)
python src\retrieval\benchmark.py

# 5. Run the full test suite
python -m pytest tests -q
```

Expect step 3 to download ~130MB once; subsequent runs use the local
Hugging Face cache (`~/.cache/huggingface` / `%USERPROFILE%\.cache\huggingface`
on Windows) and need no further network access.

**Fallback model** if `bge-small-en-v1.5` is slow or unavailable:
`sentence-transformers/all-MiniLM-L6-v2` — swap the `model_name` argument
to `LocalSentenceTransformerProvider` in `src/retrieval/benchmark.py`.

### 2. PostgreSQL/pgvector not provisioned

**Status:** Expected — Phase D (indexing/database) has not started yet
per the agreed dependency order (retrieval baseline first). Not currently
blocking anything; noting here so it isn't rediscovered as a surprise.

No Postgres instance exists in this sandbox or has been checked for
local install feasibility, since that phase hasn't started.
