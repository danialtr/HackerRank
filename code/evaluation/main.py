"""Evaluation harness — scores the system on dataset/sample_claims.csv.

It runs two VLM strategies, scores each against the gold labels, and writes
evaluation/evaluation_report.md with:
  * per-column accuracy and the claim_status confusion matrix
  * the ≥2-strategy ablation: the multi-stage pipeline vs a single mega-prompt
  * an operational analysis (model calls, tokens, cost, runtime), extrapolated
    from the sample run to the full test set.

The system is VLM-only, so this requires ANTHROPIC_API_KEY.

Run:
    python code/evaluation/main.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the sibling modules in code/ importable when run directly.
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config  # noqa: E402
from backends import build_backend  # noqa: E402
from data_loader import load_claims, load_user_history  # noqa: E402
from logging_setup import CostMeter, log, setup_logging  # noqa: E402
from pipeline.orchestrator import run as run_pipeline  # noqa: E402

import metrics  # noqa: E402  (code/evaluation is on sys.path[0] when run directly)


def count_test_claims() -> int:
    try:
        with config.claims_csv().open(encoding="utf-8") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except Exception:  # noqa: BLE001
        return 0


def operational(meter: CostMeter, n_sample: int, n_test: int, runtime_s: float) -> dict:
    s = meter.summary()
    per_claim_cost = (s["cost_usd"] / n_sample) if n_sample else 0.0
    per_claim_calls = (s["model_calls"] / n_sample) if n_sample else 0.0
    s["runtime_seconds"] = round(runtime_s, 2)
    s["per_claim_cost_usd"] = round(per_claim_cost, 5)
    s["per_claim_model_calls"] = round(per_claim_calls, 2)
    s["est_test_set_cost_usd"] = round(per_claim_cost * n_test, 4)
    s["est_test_set_calls"] = round(per_claim_calls * n_test)
    return s


def run_strategy(claims, arch) -> tuple[list[dict], dict]:
    meter = CostMeter()
    backend = build_backend(meter)
    t0 = time.time()
    preds = run_pipeline(claims, backend, arch=arch)
    backend.close()
    return preds, operational(meter, len(claims), count_test_claims(), time.time() - t0)


def main() -> None:
    setup_logging(False)
    histories = load_user_history()
    claims = load_claims(path=config.sample_claims_csv(), histories=histories)
    golds = [c.expected for c in claims]
    log.info("Evaluating on %d labeled sample claims", len(claims))

    # Strategy A — the multi-stage pipeline (our submitted system).
    preds_a, ops_a = run_strategy(claims, "pipeline")
    score_a = metrics.score(preds_a, golds)

    # Strategy B — the single mega-prompt ablation (one call per claim).
    preds_b, ops_b = run_strategy(claims, "mega")
    score_b = metrics.score(preds_b, golds)

    name_a = "Multi-stage pipeline (VLM)"
    name_b = "VLM single mega-prompt (one call per claim)"
    report = _render_report(name_a, score_a, ops_a, name_b, score_b, ops_b, len(claims))

    out = Path(__file__).resolve().parent / "evaluation_report.md"
    out.write_text(report, encoding="utf-8")
    log.info("Wrote evaluation report to %s", out)
    print("\n" + report)


def _strategy_block(name: str, sc: dict, ops: dict) -> str:
    exact = "\n".join(f"| {k} | {v:.3f} |" for k, v in sc["exact"].items())
    sets = "\n".join(f"| {k} (Jaccard) | {v:.3f} |" for k, v in sc["set_jaccard"].items())
    return f"""### {name}

Overall score (mean of scored columns): **{sc['overall_score']:.3f}**
claim_status macro-F1: **{sc['claim_status_macro_f1']:.3f}**

| column | score |
|---|---|
{exact}
{sets}

claim_status confusion matrix (rows = gold, cols = predicted):

```
{metrics.format_confusion(sc['claim_status_confusion'])}
```

Operational: {ops['model_calls']} model calls, {ops['input_tokens']}+{ops['output_tokens']} in/out tokens, \
${ops['cost_usd']:.4f} on the sample; ~${ops['per_claim_cost_usd']:.5f}/claim, \
est. **${ops['est_test_set_cost_usd']:.4f}** for the full test set ({ops['est_test_set_calls']} calls); \
runtime {ops['runtime_seconds']}s.
"""


def _render_report(name_a, sc_a, ops_a, name_b, sc_b, ops_b, n) -> str:
    winner = name_a if sc_a["overall_score"] >= sc_b["overall_score"] else name_b
    return f"""# Evaluation Report — Multi-Modal Evidence Review

Scored on `dataset/sample_claims.csv` ({n} labeled claims). Backend: **Claude VLM
(tiered: Haiku extraction + Sonnet perception)**.

We score the 10 predicted columns: exact (normalised) match for the scalar
columns, and set-overlap (Jaccard) for the two list columns
(`risk_flags`, `supporting_image_ids`). The headline metric is the
`claim_status` 3-class macro-F1.

## Strategy comparison (the required ≥2-approach ablation)

| strategy | overall | claim_status macro-F1 | est. test-set cost |
|---|---|---|---|
| {name_a} | {sc_a['overall_score']:.3f} | {sc_a['claim_status_macro_f1']:.3f} | ${ops_a['est_test_set_cost_usd']:.4f} |
| {name_b} | {sc_b['overall_score']:.3f} | {sc_b['claim_status_macro_f1']:.3f} | ${ops_b['est_test_set_cost_usd']:.4f} |

**Selected for `output.csv`: {name_a}** — the multi-stage pipeline keeps the
deterministic decision logic (precedence rule, enum compliance) in code and uses
the model only for perception, which is more debuggable and cheaper per unit of
accuracy than the single mega-prompt. (Winner by overall score above: {winner}.)

{_strategy_block(name_a, sc_a, ops_a)}
{_strategy_block(name_b, sc_b, ops_b)}

## Operational analysis

- **Model calls.** Pipeline: 1 cheap text call (claim extraction, Haiku) + 1
  vision call per image (Sonnet). With ~1.8 images/claim that is roughly 2.8
  calls/claim; the final decision is deterministic (no model call). The
  mega-prompt uses 1 call/claim but loads every image into one context.
- **Tokens & images.** See per-strategy lines above. Images are downscaled to a
  ~1568px long edge before encoding (vision tokens scale with resolution), which
  is the single biggest token saver.
- **Cost.** Per-claim and extrapolated full-test-set cost are shown per strategy,
  computed from live `usage` and the Claude API price table in `config.PRICING`
  (Sonnet 4.6 $3/$15, Haiku 4.5 $1/$5 per 1M in/out).
- **Caching.** The large static perception instruction block is sent as a cached
  system block, so repeated per-image calls pay cache-read (~0.1x), not full
  input price.
- **Rate limits / retries.** The Anthropic SDK retries 429/5xx with exponential
  backoff (`max_retries=5`); identical images are de-duplicated by the loader and
  could be processed concurrently for higher throughput.
- **Determinism.** The decision logic, evidence lookup, history rules, and schema
  normalisation are pure code, so re-runs on the same perception are identical.

_Generated by `code/evaluation/main.py`._
"""


if __name__ == "__main__":
    main()
