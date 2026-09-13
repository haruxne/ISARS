# Interval Selection for Adversarial Rank Suppression

Official implementation and reproduction materials for **“Interval Selection for
Adversarial Rank Suppression in Windowed Text-to-Video Retrieval.”**

ISARS selects a fixed-budget contiguous interval whose removal minimizes the strongest
clean retrieval score that remains outside the editable region. The selected interval is
then optimized with the same query-conditioned PGD budget used by the matched baselines.

## Repository layout

```text
configs/          Paper configuration
examples/         Small input example for interval selection
paper/            Manuscript source, bibliography, and figures
provenance/       Historical pipeline and integrity manifests
reproduction/     Reported aggregate results
scripts/          Direct Python entry points
src/isars/        Reusable method and verification code
tests/            Unit and regression tests
```

The public commands use descriptive names. Historical stage identifiers are confined to
`provenance/` because they are part of the original hash-bound evidence trail.

## Setup

Python 3.10 or newer is required.

```bash
git clone https://github.com/haruxne/ISARS.git
cd ISARS
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the optional attack implementation and development tools when needed:

```bash
python -m pip install -e ".[attack,dev]"
```

## Verify the reported results

The lightweight reproduction path requires neither the private dataset nor a GPU:

```bash
isars verify
```

or, without using the installed console command:

```bash
python scripts/verify_reported_results.py
```

This command prints the primary 93-query table and independently recomputes the exact
paired sign tests and their Bonferroni corrections. It verifies aggregate arithmetic; it
does not rerun attacks or rehash unavailable execution authorities.

## Run interval selection

Prepare a JSON object containing the clean score for every sliding window, the number of
sampled frames, and the zero-based target-frame position. A complete example is provided
in `examples/selection_input.json`.

```bash
isars select \
  --input examples/selection_input.json \
  --output outputs/selected_intervals.json
```

The output contains the matched-budget selections used in the paper:

- `target_centered`
- `highest_score_window`
- `minimum_unaffected_floor`

All candidates are contiguous and have the same frame budget. Ties for the
minimum-unaffected-floor strategy are resolved deterministically by choosing the earliest
start position.

## Full evaluation assets

The paper uses a fixed 873-video collection, promoted query records, model/cache assets,
and per-query execution authorities that are not redistributed here. Authorized users can
place those assets under an artifact root with the following semantic layout:

```text
<artifact-root>/
├── catalog/
├── frame_cache/
├── development/
└── evaluation/
    ├── round1_method_lock/
    ├── round1_evaluation/
    ├── extension_query_promotion/
    └── round2_evaluation/
```

The archival notebook in `provenance/historical_pipeline.ipynb` retains the complete
preparation, GPU evaluation, and audit implementation. Set `ISARS_ARTIFACT_ROOT` and
`ISARS_DATA_ROOT` before using its optional full-audit stages. The two GPU evaluation
stages require CUDA with at least 18 GiB of free memory.

## Reported scope

The primary result is the prospectively locked 93-query extension. The earlier 48-query
round and the pooled 141-query analysis are retained for provenance; the pooled analysis
is secondary because round-one outcomes were already known before round two was locked.

`provenance/validation.json` records what was checked during packaging. In particular,
local validation executed the aggregate reproduction and verified embedded source hashes,
but did not perform a fresh GPU replay or independently rehash the original private
artifacts.

## Citation

The citation record will be added after publication metadata is finalized. Until then,
please cite the paper title and this repository URL.

Dataset access and redistribution remain subject to the dataset owner's terms. No dataset,
model-weight, or third-party license is asserted by this repository.
