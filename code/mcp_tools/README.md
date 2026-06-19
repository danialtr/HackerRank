# MCP tools — optional orchestration demo (OFF the hot path)

The hackathon is named "Orchestrate", so this folder demonstrates how the
system's **deterministic helpers** can be exposed as Model Context Protocol (MCP)
tools that an interactive agent could call. It is intentionally **not** used by
`code/main.py` or the evaluation harness.

## Why it is not on the critical path

This task is a **batch ETL job**: read N rows, run the same fixed pipeline on
each, write a CSV. The control flow is known in advance — we always check image
quality, always look up the evidence requirement, always read history. MCP's
value is letting a model *dynamically choose* which tool to call in an open-ended
session; here there is no such choice to make. An agentic MCP loop would also add
reasoning tokens and round-trips, working against the operational analysis the
challenge explicitly rewards (fewer calls, fewer tokens).

So MCP is included as a **considered-and-kept-off-the-hot-path** design choice:
it shows we can wrap our tools for an agent, while the substance (a deterministic
pipeline calling a VLM only for perception) stays simple, cheap, and debuggable.

## What it exposes

`server.py` (a [FastMCP](https://modelcontextprotocol.io) server) exposes three
deterministic tools, each a thin wrapper around code already used by the pipeline:

| MCP tool | Wraps | Returns |
|---|---|---|
| `assess_image_quality` | `backends/heuristic.py` CV stats | blur/brightness + quality flags for an image |
| `lookup_evidence_requirement` | `pipeline/evidence.py` | the minimum-evidence rule for (object, part, issue) |
| `get_user_history` | `data_loader.py` | the user-history risk row for a user_id |

## Run (optional)

```bash
pip install mcp            # only needed for this demo
python code/mcp_tools/server.py
```
