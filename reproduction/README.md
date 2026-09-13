# Reported-result verification

`reported_results.json` contains the aggregate output of the locked 93-query extension
and the secondary pooled analysis. It does not contain per-query execution authorities.

Run the independent arithmetic checks from the repository root:

```bash
python -m pip install -e .
isars verify
```

The command validates cohort denominators and recomputes the exact paired sign tests with
the recorded Bonferroni family size. It does not claim a new inference run, GPU replay, or
independent verification of the private source artifacts.
