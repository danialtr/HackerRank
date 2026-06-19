"""Deterministic pipeline stages: evidence sufficiency, history risk, fusion.

The VLM backend is the "eyes" (perception). Everything in this package is the
"adjudicator" — plain code that applies the rulebook, so the precedence rule
(images beat history) and enum compliance are guaranteed, not hoped for.
"""
