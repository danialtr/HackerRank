# Multi-Modal Evidence Review — `code/`

Verifies damage claims (car / laptop / package) by reading the claim
conversation, the submitted images, user history, and the minimum evidence
requirements, then producing a structured 14-column decision per claim.

The system is **VLM-only**: perception is done by Claude vision models, and the
decision is made by deterministic code. It requires `ANTHROPIC_API_KEY`.

For the full design write-up (what each part does, why, the models, frameworks,
and libraries) read [`SOLUTION.md`](./SOLUTION.md).

## Layout

```
code/
├── main.py              # entry point: dataset -> output.csv
├── run-sample.py        # run one or a few CHOSEN claims (testing)
├── config.py            # paths, model tiering, pricing, runtime switches
├── schema.py            # enums + column order + validator/normaliser
├── logging_setup.py     # runtime logging + token/cost meter
├── models.py            # typed inputs + pipeline result objects
├── data_loader.py       # stage 1: load CSVs + probe images, join history
├── image_utils.py       # decode / downscale / probe images
├── retrieval.py         # dynamic few-shot from sample_claims.csv
├── backends/            # the "eyes" (perception)
│   ├── base.py          #   PerceptionBackend interface
│   └── vlm.py           #   Claude VLM backend (tiered, structured output)
├── pipeline/            # the deterministic "adjudicator"
│   ├── evidence.py      #   stage 6: evidence sufficiency
│   ├── history_risk.py  #   stage 7: user-history risk (gated)
│   ├── fuse.py          #   stage 8: final decision
│   └── orchestrator.py  #   runs all stages, builds normalised rows
├── evaluation/
│   ├── main.py          # scores the 2-strategy ablation vs sample_claims.csv
│   ├── metrics.py       # accuracy, confusion matrix, Jaccard
│   └── evaluation_report.md   # generated report
└── logs/                # per-run logs (run_<timestamp>.log)
```

## Setup

```bash
pip install -r code/requirements.txt
```

The API key comes from the environment only (never hardcoded). Two ways to set it:

```bash
# Option 1 — export it in your shell (one-off):
export ANTHROPIC_API_KEY=sk-ant-...

# Option 2 — a .env file at the repo root (auto-loaded by config.py, gitignored):
cp .env.example .env        # then edit .env and paste your key
```

A key is **required** — the system is VLM-only and exits with a clear message if
no key is set. An exported variable overrides `.env`.

## Run

```bash
python code/main.py                       # test split -> output.csv
python code/main.py --split sample        # run on the labeled sample split
python code/main.py --arch mega           # single mega-prompt ablation
python code/main.py -v                     # full per-stage debug trace
python code/main.py --limit 3             # only the first 3 claims (quick test)
```

## Test individual claims

```bash
python code/run-sample.py --list             # list selectable claims
python code/run-sample.py --ids user_005     # one claim (pred vs gold on sample)
python code/run-sample.py --index 1,4,7      # several by row number
python code/run-sample.py --case case_008    # by image case folder
```

## Evaluate

```bash
python code/evaluation/main.py             # writes evaluation/evaluation_report.md
```

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **required** — enables the Claude VLM backend |
| `EVR_ARCH` | `pipeline` | `pipeline` (multi-stage) / `mega` (single call) |
| `EVR_MODEL_EXTRACT` | `claude-haiku-4-5` | text model for claim extraction |
| `EVR_MODEL_PERCEPTION` | `claude-sonnet-4-6` | vision model for per-image perception |
| `DATASET_DIR` | `<repo>/dataset` | dataset location |
| `OUTPUT_CSV` | `<repo>/output.csv` | predictions path |
