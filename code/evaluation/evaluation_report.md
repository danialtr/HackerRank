# Evaluation Report — Multi-Modal Evidence Review

> This report is **generated** by the evaluation harness, which requires an API
> key (the system is VLM-only). It has not been regenerated in this environment
> because no key was available. To produce it:
>
> ```bash
> export ANTHROPIC_API_KEY=sk-ant-...      # or use a .env file
> python code/evaluation/main.py
> ```

The harness scores the system on `dataset/sample_claims.csv` (20 labeled claims)
and writes the full results here, including:

- **Per-column accuracy** for the scalar columns and **Jaccard** (set overlap)
  for the list columns (`risk_flags`, `supporting_image_ids`).
- The **`claim_status` 3-class confusion matrix** and **macro-F1** (the headline
  metric).
- The required **≥2-strategy ablation**: the multi-stage pipeline (Haiku
  extraction + Sonnet per-image perception + deterministic fusion) versus a
  single **mega-prompt** (one call per claim).
- An **operational analysis**: model calls, input/output tokens, images
  processed, per-claim and extrapolated full-test-set cost (from live `usage` and
  `config.PRICING`), runtime, plus the caching / retry strategy.
