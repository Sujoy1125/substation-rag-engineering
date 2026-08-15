# Setup — running this project on your machine

Aimed at a teammate who has just received the project and wants it running.
Five minutes if nothing goes wrong; `scripts/check_setup.py` tells you what
went wrong if something does.

---

## 1. Get the code

**Preferred — clone it.** You get the history, you can pull updates, and you
cannot accidentally receive somebody's API key:

```
git clone https://github.com/Sujoy1125/substation-rag-engineering
cd substation-rag-engineering
```

**If you were sent a ZIP**, check it before trusting it:

- It must contain `KB_v1.1/KB_v1.1_final/knowledge_chunks.xlsx` (~400 KB). The
  knowledge base is the project; without it nothing runs.
- It must **not** contain a `venv/` folder. A virtual environment records
  absolute paths from the machine that made it and will not work on yours.
- It must **not** contain a `.env` file. That file holds an API key. If one
  arrives in your ZIP, tell whoever sent it — the key is now on every machine
  that received the archive and should be revoked.

> **If you are the one making the ZIP:** do not zip the project folder from
> your file manager. It will include `venv/` (tens of thousands of files) and
> `.env` (your API key). Use `git archive` instead, which packages exactly what
> is committed and nothing else:
>
> ```
> git archive --format=zip --output=../substation-rag.zip HEAD
> ```

---

## 2. Create a virtual environment

A virtual environment keeps this project's packages separate from everything
else on your machine. It is not optional in practice — without one, installing
here can break another project.

```
python -m venv venv

venv\Scripts\activate            # Windows (PowerShell / CMD)
source venv/bin/activate         # macOS / Linux
```

Your prompt should now start with `(venv)`. It must say that every time you
work on the project — activation does not persist between terminal sessions.

**Python 3.10 or newer is required.** Check with `python --version`.

---

## 3. Install the dependencies

```
pip install -r requirements.txt
```

About a minute, roughly 100 MB. That is everything the project actually runs
on: knowledge base loading, retrieval, generation, the web service and the
tests.

**Do not install `requirements-dense.txt` unless you are specifically working
on dense retrieval.** It is a ~2.5 GB download (PyTorch and
sentence-transformers) for a code path that is unmeasured and blocked — see
`docs/BLOCKERS.md`. Nothing else needs it: the import is lazy, and the three
tests that use it skip themselves when it is absent.

---

## 4. Check the setup

```
python scripts/check_setup.py
```

Seven checks, in dependency order, and it never stops at the first failure —
knowing that three things are wrong is more useful than being told about the
first one three times. Each failure names the fix rather than the symptom.

It makes no paid API call.

---

## 5. Run the tests

```
python -m pytest -q
```

Expect **198 passed** (or 194 passed, 4 skipped if you did not install the
optional dense dependencies — that is correct, not a problem).

Every test runs offline against a scripted client: no API key, no network, no
cost. They do not test whether the language model answers well; they test that
the code around it cannot be fooled.

---

## 6. Run it

```
uvicorn src.api.service:build_default_app --factory --reload
```

Then open <http://127.0.0.1:8000/> for the interface, or
<http://127.0.0.1:8000/docs> for the API.

### Without an API key

Most of the system still works, and this is worth understanding rather than
treating as a degraded mode:

| Works without a key | Needs a key |
|---|---|
| Retrieval over all 1,745 chunks | Generating an answer |
| `GET /evidence/{chunk_id}` — the source records | `POST /ask` |
| `GET /facets` — knowledge base coverage | The evaluation runs |
| `GET /health`, the whole interface shell | |
| `python scripts/ask.py "..." --evidence` | |
| The full test suite | |

`POST /ask` returns **503** with the specific cause named — out of credit,
rate limited, bad key, or network. That is deliberate: "the provider is
unavailable" and "this service is broken" need different reactions.

---

## 7. Add an API key (only if you need generation)

```
copy .env.example .env           # Windows
cp .env.example .env             # macOS / Linux
```

Then open `.env` and set `OPENAI_API_KEY`.

> **The key goes in `.env`, never in `.env.example`.**
> `.env` is gitignored; `.env.example` is committed. A key committed once is
> compromised permanently, even after you delete it, because it stays in the
> git history. This has already happened to this project once and the key had
> to be revoked.

**Leave optional keys commented out, not blank.** `OPENAI_BASE_URL=` with an
empty value is not the same as unset — the OpenAI SDK reads it directly and
treats `""` as "configured, to nothing", producing a bare "Connection error"
that looks exactly like a firewall. This cost several hours of debugging. The
loader now skips empty values too, so both halves are protected.

### Using a provider other than OpenAI

The client speaks plain OpenAI protocol, so any compatible endpoint works with
**no code change** — three values in `.env` and `LLM_PROVIDER` left as
`openai`. See the comments in `.env.example` for Azure OpenAI, Google Gemini,
Groq and OpenRouter, including their rate limits.

**Do not change model between the calibration run and the holdout run.** The
confidence weights are fitted per model.

---

## 8. If generation fails

```
python scripts/diagnose_network.py
```

Ten steps, from configuration through DNS, TCP, TLS and authentication to a
real one-token generation. It exists because "Connection error" from the SDK
covers at least six distinct failures with six different fixes.

Step 10 matters most: **listing models is free and generating is not**, so a
key can pass every connectivity check and still fail on the first real call.
If step 10 reports 429 with a quota code, the account has no credit and no
code change can work around it.

---

## Common problems

**`ModuleNotFoundError: No module named 'src'`** — you are not in the project
root. `cd` to the folder containing `requirements.txt`.

**`python` is not recognised (Windows)** — try `py` instead of `python`, or
reinstall Python with "Add to PATH" ticked.

**Tests fail with `No module named 'fastapi'`** — the virtual environment is
not activated. Look for `(venv)` in your prompt.

**`.git/index.lock` errors** — a previous git command was interrupted. Delete
`.git/index.lock` and retry.

**Everything installs but retrieval finds nothing** — the knowledge base
workbook is missing or truncated. Run `python scripts/check_setup.py`; step 5
will say so. Re-clone rather than repairing by hand.
