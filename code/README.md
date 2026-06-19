# Multi-Modal Evidence Review — `code/`

Verifies damage claims (car / laptop / package) by reading the claim
conversation, the submitted images, user history, and the minimum evidence
requirements, then producing a structured 14-column decision per claim.

For the full design write-up (what each part does, why, the models, frameworks,
and libraries) read [`SOLUTION.md`](./SOLUTION.md).

## Layout

```
code/
├── main.py              # entry point: dataset -> output.csv
├── config.py            # paths, model tiering, pricing, runtime switches
├── schema.py            # enums + column order + validator/normaliser
├── logging_setup.py     # runtime logging + token/cost meter
├── models.py            # typed inputs + pipeline result objects
├── data_loader.py       # stage 1: load CSVs + probe images, join history
├── image_utils.py       # decode / downscale / probe images
├── retrieval.py         # dynamic few-shot from sample_claims.csv (VLM only)
├── backends/            # the swappable "eyes"
│   ├── base.py          #   PerceptionBackend interface
│   ├── vlm.py           #   Claude VLM backend (tiered, structured output)
│   └── heuristic.py     #   deterministic CV + claim-text fallback
├── pipeline/            # the deterministic "adjudicator"
│   ├── evidence.py      #   stage 6: evidence sufficiency
│   ├── history_risk.py  #   stage 7: user-history risk (gated)
│   ├── fuse.py          #   stage 8: final decision
│   └── orchestrator.py  #   runs all stages, builds normalised rows
├── evaluation/
│   ├── main.py          # scores ≥2 strategies vs sample_claims.csv
│   ├── metrics.py       # accuracy, confusion matrix, Jaccard
│   └── evaluation_report.md   # generated report
├── mcp_tools/           # optional MCP demo (OFF the hot path)
└── logs/                # per-run logs (run_<timestamp>.log)
```

## Setup

```bash
pip install -r code/requirements.txt
# secrets come from env vars only:
export ANTHROPIC_API_KEY=sk-...     # optional — enables the VLM backend
```

If no key is set the system automatically uses the deterministic **heuristic**
backend, so it always runs end to end.

## Run

```bash
python code/main.py                       # test split -> output.csv
python code/main.py --split sample        # run on the labeled sample split
python code/main.py --backend heuristic   # force the deterministic backend
python code/main.py --backend vlm         # force the Claude VLM backend
python code/main.py --arch mega           # single mega-prompt ablation (VLM)
python code/main.py -v                     # full per-stage debug trace
python code/main.py --limit 3             # only the first 3 claims (quick test)
```

## Evaluate

```bash
python code/evaluation/main.py             # writes evaluation/evaluation_report.md
python code/evaluation/main.py --backend heuristic
```

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | enables the VLM backend |
| `EVR_BACKEND` | `auto` | `auto` / `vlm` / `heuristic` |
| `EVR_ARCH` | `pipeline` | `pipeline` (multi-stage) / `mega` (single call) |
| `DATASET_DIR` | `<repo>/dataset` | dataset location |
| `OUTPUT_CSV` | `<repo>/output.csv` | predictions path |
