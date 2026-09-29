# Data and derived-input contract

The raw third-party traces are not included. Do not redistribute them until their
licences and the authors' rights have been checked. For each trace, the public
release must give its original landing page and citation, version/download date,
access terms, exact raw-file SHA-256, filtering and causal-smoothing steps, and the
processed-file SHA-256. Official *candidate* upstream pages are now identified
below; the mapping from each local processed file to a specific upstream
download/version and the right to redistribute derived files remain unresolved.

| Input | Code location expected | Source/version/licence | Status |
| --- | --- | --- | --- |
| Alibaba trace | `prediction_experiments/FITSTest/usage_2018.csv` | Official `cluster-trace-v2018` documentation; exact local-file lineage and redistribution terms unverified | Not staged |
| GWA-T-12 trace | `prediction_experiments/FITSTest/data1.csv` | Official BitBrains GWA-T-12 page; local fastStorage/Rnd lineage unverified | Not staged |
| Google trace | `prediction_experiments/FITSTest/data2.csv` | Official Google cluster-data repository; specific trace version used locally unverified | Not staged |
| Aligned resource-state series | `raft/test/value.csv` | Derived from formal prediction/export pipeline | Exact formal file identified by hash below, not staged |
| Five-step CPU predictions | `raft/test/cpu_predictions.csv` | Same formal export | Exact formal file identified by hash below, not staged |
| Five-step memory predictions | `raft/test/memory_predictions.csv` | Same formal export | Exact formal file identified by hash below, not staged |

## Official upstream pages located (not yet local-file provenance)

- [Alibaba Cluster Trace v2018 documentation](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md) identifies `machine_usage.csv` and the upstream download/checksum route. The local two-column `usage_2018.csv` has not been proven byte-for-byte or transformation-equivalent to a particular upstream release. The page's research-access description does not by itself establish permission to mirror the processed file.
- [GWA-T-12 BitBrains](https://atlarge-research.com/gwa-t-12/) provides separate fastStorage and Rnd downloads and requires acknowledgement of Bitbrains in publications using the trace. Which source files and transformations produced local `data1.csv` still needs to be documented; a separate redistribution licence was not established here.
- [Google cluster-data](https://github.com/google/cluster-data) describes several trace versions and states that its data and documentation are available under CC-BY. The local `data2.csv` has not yet been mapped to a specific Google trace version or upstream record, so its reuse statement must remain conditional on that mapping and attribution.

The local raw files were hashed on 2026-09-29. These are fingerprints of the
workspace bytes, **not** verification that they match a named upstream release:

| Local raw file | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `usage_2018.csv` | 1,958,368,499 | `8e711553498b2b009d448bd2baea37074d993fe6bc5d6864c2470e01d60d0d86` |
| `data1.csv` | 108,821,710 | `f09efc24bf3b2efffd0e0e64d14471d08d213dfc821d855a11ab1d0d61358938` |
| `data2.csv` | 48,436,791 | `bd7b477aff23b8e0d081d6826f5686253b829f606a3aec84b95b995bde2881e5` |

## Verified local Alibaba projection, upstream archive still unresolved

A read-only full-file comparison using
`scripts/verify_alibaba_usage_projection.py` checked **246,934,820** aligned
data rows of the local `machine_usage.csv` against `usage_2018.csv`. The latter
is the former's zero-based CPU and memory columns 2 and 3, in the same row
order, with numeric serialization differences (for example `41` versus
`41.0`). Six raw CPU cells are blank; every corresponding derived cell is
filled with `38.16611798205173`, matching the mean of observed raw CPU values
to the checker's `1e-6` tolerance. No other row or column-value mismatch was
found. The local full `machine_usage.csv` SHA-256 is
`ca89f90e458f33277604e12b11891e389127277036e90a823e0d26c2017cf34b`;
the local derived `usage_2018.csv` SHA-256 is listed above.

This establishes the **local transformation relationship**, not the identity
of the original download. The [official Alibaba v2018 documentation](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)
publishes SHA-256 `3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a`
for `machine_usage.tar.gz`, but that archive is not present here to verify
against the extracted local CSV. Official-version linkage, download date and
redistribution permission remain open.

The formal 100-node **original-MPC-DP** run metadata records these SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `value.csv` | `e2bf44333cd9c11cd15826f8a256c8928847e01ce72e82d900a54ddb281e845d` |
| `cpu_predictions.csv` | `64388fb14c0a53274d1ff31ae7444087ea137c42fa214dc09448beea3c5e06b9` |
| `memory_predictions.csv` | `920a65d6416b8fdf604d15ae1dfd147014566034560af9d6df5940d2aaa52a04` |

Matching files were found both in the archived
`prediction_formal_unified_v3/simulation_inputs/fits_delta_seed_2026` output and
the `fits_delta_unified_20260914` industrial-PC provenance bundle. The generator
is `prediction_experiments/baseline_experiments/prepare_formal_prediction_bundle.py`; its
archived `trace_manifest.json` records the Alibaba source hash above, first
100,000 rows, chronological 70/15/15 split, five-point causal smoothing,
100-step input, five-step horizon, and FITS-Delta seed-2026 checkpoint hash
`b24c06996f06a221cde176e1be732902775f542dad3eabac655b4ab31de0d43f`.
That manifest contains local absolute paths and would need sanitization before
any public distribution. Exact generated CSVs and checkpoints are retained only
in the authors' separate archive and are not distributed in this repository.
Any later redistribution requires separate authorization.
Two local Alibaba seed-2026 FITS-Delta checkpoint archives match its recorded
checkpoint SHA-256, and all 20 final-testbed node evidence records report the
same deployed weight hash. This is model-version provenance, not a sharing
permission or an independent training reproduction.
The same filenames currently under the workspace's `raft/test/` have different
hashes and must not be silently substituted. The formal 100-node run used
`trace_length=282`, 25 network instances, 120 requests per network, a 50% update
budget and `threshold_lambda=0`; the network, not each request, is the replicate.

The read-only `scripts/verify_formal_prediction_inputs.py INPUT_DIRECTORY`
checks the frozen three-file SHA-256 values, headers, numeric finiteness and
CSV shapes, and checks that both `step_0` columns agree row by row with
`value.csv`. The concise file-level dictionary, sizes, derivation and
machine-readable manifest are in [`formal_inputs/README.md`](formal_inputs/README.md)
and [`formal_inputs/manifest.json`](formal_inputs/manifest.json). The CSV bytes
are not distributed in this repository because redistribution permission has
not been confirmed.
The verifier passed against the archived
`prediction_formal_unified_v3/simulation_inputs/fits_delta_seed_2026` export:
14,896 two-column resource-state rows and 14,896 six-column prediction rows
for each resource. This verifies the downstream input files, not upstream
dataset rights or a fresh FITS-Delta training run.

## Rebuild the three formal CSVs from completed prediction outputs

The included `prepare_formal_prediction_bundle.py` accepts a local
`--alibaba-csv` path and a `--trace-only` mode; industrial-PC timing records
are not needed for this route. It still requires a completed nine-model
prediction output directory with the seed-2026 FITS and FITS-Delta checkpoints
and saved predictions. The trained weights have **not** been cleared for public
release; a reader must first regenerate those outputs from an approved trace
copy or obtain an authorized checkpoint separately.

```sh
python -B prediction_experiments/baseline_experiments/prepare_formal_prediction_bundle.py \
  --trace-only \
  --alibaba-csv PATH_TO_LOCAL_USAGE_2018_CSV \
  --benchmark-dir PATH_TO_COMPLETED_NINE_MODEL_OUTPUT \
  --output PATH_TO_NEW_REBUILT_BUNDLE
python -B scripts/verify_formal_prediction_inputs.py \
  PATH_TO_NEW_REBUILT_BUNDLE/simulation_inputs/fits_delta_seed_2026
```

On 2026-09-29 this trace-only command was run against the locally archived
completed benchmark and locally verified Alibaba projection. Its three
FITS-Delta CSVs matched the formal hashes above **byte-for-byte**, each with
14,896 rows. This checks the local reconstruction path, not a clean-machine
training run or a third-party redistribution permission. The CSV schema is:
`value.csv` has no header and holds current CPU and memory utilization;
`cpu_predictions.csv` and `memory_predictions.csv` have `step_0` through
`step_5`, where step zero is the current observation and steps one to five are
forecasts. Values are utilization percentages, not fractions.

For a future dataset deposit, provide a data dictionary with column names, units,
missing-value handling, trace selection, transformations, and a file-to-table/figure
map. Publish third-party access instructions when redistribution is prohibited.
