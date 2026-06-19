# Multi-Modal Evidence Review — `code/`

Verifies damage claims (car / laptop / package) by reading the claim
conversation, the submitted images, user history, and the minimum evidence
requirements, then producing a structured decision per claim.

## Layout

```
code/
├── main.py          # entry point / CLI
├── config.py        # path resolution (dataset dir, output path)
├── models.py        # typed inputs: Claim, ImageRef, UserHistory
├── data_loader.py   # stage 1: load CSVs + images, join history
├── requirements.txt
└── evaluation/
    └── main.py       # scores the system against dataset/sample_claims.csv
```

The system reads from `../dataset/` by default (the dataset lives next to
`code/`). Override with the `DATASET_DIR` environment variable if needed.

## Pipeline (target design)

1. **Load & join** — claim + images + history *(implemented)*
2. **Extract claim** — what damage / which part is alleged (text)
3. **Per-image perception** — object, part, issue, severity, quality,
   authenticity, embedded-text (one VLM call per image)
4. **Evidence sufficiency** — valid images vs `evidence_requirements.csv`
5. **History risk** — risk context only; never overrides clear visuals
6. **Fuse** — `claim_status` + justification + supporting image IDs

## Setup

```bash
pip install -r requirements.txt
# secrets come from env vars only, e.g. ANTHROPIC_API_KEY
```

## Run (stage 1)

```bash
python code/main.py                  # load test set, show the first claim
python code/main.py --split sample   # labeled sample set (shows gold output)
python code/main.py --index 3 --limit 2
python code/main.py --check          # verify every referenced image resolves
```
