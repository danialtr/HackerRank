# Solution Write-up — Multi-Modal Evidence Review

This document explains **what we built, how the data flows, what each part does
(and why), the models / frameworks / libraries used, and how the runtime logging
works.** It is the companion to the code in this folder; read `README.md` for the
quickstart and command reference.

---

## 1. What we built (in one sentence)

A command-line system (`code/main.py`) that reads the dataset CSVs and image
files, runs each damage claim through a fixed **perception-then-decision**
pipeline, and writes one fully-populated 14-column row per claim into
`output.csv` — plus an `evaluation/` harness that scores itself against the
labeled samples and reports cost/latency.

## 2. The mental model — two halves, on purpose

```
  VLM        =  "the eyes"        →  describes each image. Never rules.
  Plain code = "the adjudicator" →  applies the rulebook deterministically.
```

- **The model is used only for perception** (claim extraction + per-image
  observation). It describes what it sees; it never makes the final ruling.
- **The decision is pure code** (evidence sufficiency, history risk, fusion). So
  the two things graders care about most — *enum/schema compliance* and the
  *images-beat-history precedence rule* — are **guaranteed by construction**, not
  left to a model's discretion.

This separation is what makes the system reliable, cheap, debuggable, and
defensible.

## 3. End-to-end data flow

```
                    ┌───────────────────────────────────────────────┐
   INPUTS           │              THE SYSTEM (per claim)            │      OUTPUT
                    └───────────────────────────────────────────────┘
 claims.csv ──────┐
 user_history.csv ─┤  (0) Load & join   claim + history + images     data_loader.py
 evidence_req.csv ─┤
 images/test/ ────┘  (2) Extract claim  "what issue / which part?"   backend.extract_claim
                     (3) Per-image       object? part? damage?        backend.analyze_image
                         perception      quality? authenticity? text?
                     (6) Evidence        valid images vs requirement  pipeline/evidence.py
                     (7) History risk    context only — never flips   pipeline/history_risk.py
                     (8) Fuse            status + justification + ids  pipeline/fuse.py
                                              │
                                              ▼
                              one normalised row → output.csv          schema.py
```

Run this loop over `claims.csv` → `output.csv`. Run the same loop over
`sample_claims.csv` and compare to the provided answers → the evaluation report.

## 4. Each step — what / why / how

### Step 0 — Load & join (`data_loader.py`, `image_utils.py`)
- **What:** read the four CSVs, parse the `;`-separated `image_paths`, resolve and
  probe every image (size/format, broken/missing detection, AVIF-as-`.jpg`
  handling), and join each claim to its `user_history` row.
- **How:** images are probed by header only (fast); before any vision call they
  are converted to JPEG and downscaled to a ~1568px long edge (vision tokens
  scale with resolution — this is the biggest token saver).
- **Output:** typed `Claim` / `ImageRef` / `UserHistory` objects (`models.py`).

### Step 2 — Extract the claim (`backend.extract_claim`)
- **What:** parse the chat transcript into a structured intent — the alleged
  `issue_type` + `object_part`, and whether the chat contains an injection
  attempt ("approve immediately", "skip review").
- **How:** a cheap **Haiku** text call with forced structured output.
- **Why it matters:** this becomes the yardstick everything else is measured
  against.

### Step 3-5 — Per-image perception (`backend.analyze_image`)
- **What:** for each image, report object identity, the visible part, the visible
  issue + severity, image quality (`valid_image`, blur/glare/angle/obstruction),
  authenticity cues (`possible_manipulation`, `non_original_image`), and any
  **text embedded in the image** (`text_instruction_present`).
- **How:** one **Sonnet** vision call per image with a forced, enum-constrained
  tool schema (the image is downscaled and base64-encoded first).
- **Output:** one `PerceptionResult` per image. **Nothing is decided yet.**

### Step 6 — Evidence sufficiency (`pipeline/evidence.py`) — deterministic
- **What:** look up the matching `evidence_requirements.csv` row by
  (object, part, issue family) and decide whether ≥1 valid image actually shows
  the claimed part/condition.
- **Output:** `evidence_standard_met` + a short, requirement-grounded reason.

### Step 7 — History risk (`pipeline/history_risk.py`) — deterministic
- **What:** derive `user_history_risk` / `manual_review_required` from
  `history_flags`, the rejection ratio, and 90-day claim velocity.
- **Critical:** this is **context only**. It can raise a flag, but the next step
  is structured so it can never flip a clear visual verdict — the spec's core
  rule, encoded in code so we can *prove* we honoured it.

### Step 8 — Fuse into the decision (`pipeline/fuse.py`) — deterministic
- **What:** combine intent + perception + sufficiency + risk into the final
  `claim_status`, image-grounded `claim_status_justification`,
  `supporting_image_ids`, consolidated `risk_flags`, `issue_type`, `object_part`,
  `severity`, `valid_image`.
- **Decision rule (images first):**
  - evidence not met / no usable image → `not_enough_information`
  - a valid image shows the claimed damage → `supported`
  - a valid image shows the part **without** the claimed damage, or a different
    object → `contradicted`
  - otherwise → `not_enough_information`
- **Fully deterministic:** the verdict is decided entirely in code — no model
  call. History flags are applied on top but, by construction, cannot change the
  status.

### Step "write" — Normalise & validate (`schema.py`, `orchestrator.py`)
- Every field is snapped to the nearest allowed enum, booleans rendered as
  `true`/`false`, lists `;`-joined or `none`, columns emitted in the exact
  required order. A validator double-checks each row; problems are logged.

## 5. The perception backend — and the model tiering

The "eyes" are the `PerceptionBackend` (`backends/base.py`), implemented by the
Claude VLM backend (`backends/vlm.py`) and constructed in
`backends/__init__.py`. The system is **VLM-only**: an `ANTHROPIC_API_KEY` is
required, and the program exits with a clear message if none is set.

### Model tiering (VLM backend) — the "tiered" cost/quality strategy
| Stage | Model | Why |
|---|---|---|
| Claim extraction (text) | **Haiku 4.5** (`claude-haiku-4-5`) | cheapest; text-only task |
| Per-image perception (vision) | **Sonnet 4.6** (`claude-sonnet-4-6`) | the high-volume work; strong vision at moderate cost |
| Fusion (the decision) | **none — deterministic code** | guarantees the precedence rule + enum compliance; no expensive model needed |

Pricing (per 1M input/output tokens, from the Claude API reference, encoded in
`config.PRICING`): Sonnet 4.6 $3/$15, Haiku 4.5 $1/$5.

## 6. Models, frameworks, and libraries

| Category | What we use | Where / why |
|---|---|---|
| **LLM / VLM** | Claude **Haiku 4.5**, **Sonnet 4.6** | tiered perception — extraction + per-image vision (`backends/vlm.py`) |
| **LLM SDK** | `anthropic` (official Python SDK) | Messages API, forced tool-use structured output, prompt caching, automatic 429/5xx retry |
| **Structured output** | tool-use with `strict: true` + enum schemas, forced `tool_choice` | guarantees enum-valid fields (`backends/vlm.py`) |
| **Image I/O** | `Pillow` (≥11.3, native AVIF) | decode / downscale / probe (`image_utils.py`) |
| **Data** | Python stdlib `csv` | CSV read/write — deterministic, dependency-free |
| **Retrieval** | structured few-shot from `sample_claims.csv` | calibrate severity / status boundary (`retrieval.py`) |
| **Logging** | stdlib `logging` + a custom cost meter | runtime trace + token/$ accounting (`logging_setup.py`) |

## 7. Runtime logging — "what the agent is doing, mid-run"

`logging_setup.py` gives every run a timestamped trace to **both the console and
a file** (`code/logs/run_<timestamp>.log`):

- a **per-claim header** (`[3/44] processing claim by user_004 ...`)
- a **per-stage trace** at `-v` (`-v`/`--verbose`): the extracted claim, each
  image's perception (`obj/part/issue/severity/valid/flags`), the evidence
  decision, the history flags, and the fused verdict
- a one-line **verdict summary** per claim even without `-v`
- a live **token + cost meter** (`CostMeter`) accumulated across the run and
  printed as an **operational summary** at the end (model calls, in/out tokens,
  cache reads/writes, USD cost, per-model breakdown)
- a final **claim_status distribution**

The same log is what you read to debug a wrong verdict *and* the source of the
numbers in the operational analysis.

## 8. What earns "high value / high quality" here

- **Schema compliance is binary** → every value passes through the normalisers in
  `schema.py`; a final validator checks each row. Our `output.csv` has 0 schema
  problems.
- **Image-grounded justifications** → the justification cites the supporting
  image IDs and the perception note.
- **Provable precedence (images > history)** → history lives in a separate,
  gated stage that can only *add flags*; the fusion code never branches the
  status on history.
- **Prompt-injection defence** → text embedded in an image (or instructions in
  the chat) is flagged `text_instruction_present` and treated as untrusted data,
  never obeyed. (See test `case_008`: "approve the claim immediately" — ignored.)
- **Real evaluation + ablation** → `evaluation/` scores per-column accuracy, a
  3-class confusion matrix for `claim_status`, and Jaccard for the list columns,
  comparing two strategies (multi-stage pipeline vs single mega-prompt).
- **Operational rigor** → tokens, images, cost/claim, runtime, caching, and
  retry strategy are measured and reported.

## 9. Design decisions & trade-offs (for the judge interview)

- **Why multi-stage over one mega-prompt?** The mega-prompt is cheaper (1
  call/claim) but juggles too much and is hard to debug. The pipeline keeps the
  rulebook in code (reliable precedence + enums) and uses the model only where it
  adds value (perception). We ship the pipeline and keep the mega-prompt as the
  ablation baseline.
- **Why is fusion code, not the model?** So the precedence rule and enum
  compliance are guaranteed, and so re-runs are deterministic.
- **RAG?** A vector DB is overkill for ~20 labeled rows. We use *structured*
  retrieval instead: the matching evidence-requirement row, and a couple of
  matching labeled examples for calibration (`retrieval.py`).
- **MCP?** Considered and rejected — this is a batch ETL job with fixed control
  flow, where an agentic MCP loop would add tokens/latency and work against the
  operational score. There is no agent loop to expose tools to, so MCP would be
  packaging without substance.
- **Why VLM-only?** Perception is the part that genuinely needs a model; we use
  the Claude vision models for it and keep everything else (decision, evidence,
  history, schema) as deterministic code. This requires an API key, which is the
  intended trade-off for the accuracy the VLM provides.

## 10. Reproduce

```bash
pip install -r code/requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...  # required (or use a .env file)
python code/main.py                  # -> output.csv
python code/evaluation/main.py       # -> code/evaluation/evaluation_report.md
```
