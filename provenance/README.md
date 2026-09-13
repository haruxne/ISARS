# Provenance archive

This directory preserves the historical, hash-bound evaluation pipeline used to produce
the reported results. It is intentionally separate from the public `isars` package and
command-line interface.

- `historical_pipeline.ipynb` contains the preparation, evaluation, and audit stages.
- `source_manifest.json` binds the embedded runtime and adapted stage sources.
- `validation.json` states exactly which packaging checks were executed.

Storage roots and timeline-based working folders were replaced with semantic aliases.
Stage identifiers and evidence hashes remain unchanged because they are part of the
original provenance chain.

The archive does not contain raw videos, model weights, caches, credentials, human-review
records, or per-query execution authorities.
