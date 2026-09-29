# Formal prediction inputs: release metadata

**Status:** metadata only. The three CSVs remain in the authors' private
review area until permission to redistribute these Alibaba-derived files has
been confirmed. This directory is not a data deposit, and no model checkpoint
is included. Do not substitute same-named files from another experiment.

## Source and construction

The source is the CPU and memory utilization columns of Alibaba's
[`cluster-trace-v2018` machine-usage trace](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md).
The [official schema](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt)
names these fields `cpu_util_percent` and `mem_util_percent`. The local
two-column projection used by the formal run is `usage_2018.csv` (SHA-256
`8e711553498b2b009d448bd2baea37074d993fe6bc5d6864c2470e01d60d0d86`).
Its local transformation relationship to the extracted `machine_usage.csv`
was checked, but the original downloaded archive was unavailable, so an exact
match to a particular official archive is **not** established. See
[`../README.md`](../README.md) for the upstream checksum and that caveat.

The code uses the first 100,000 rows in their retained order. Numeric missing
values are linearly interpolated in both directions, followed by a five-point
causal moving average (`min_periods=1`) and float32 conversion. The processed
array SHA-256 is
`e8b9fd913bd0b9d09473e1456ff655bd3369e7b98d2709c2f2da17397f0c5e5b`.
The split is chronological: 70,000 training rows, 15,000 validation rows,
and 15,000 test rows. Each held-out window has 100 input slots and five future
slots. The export uses the last value of each input window as the current
observation and the saved seed-2026 FITS-Delta forecasts for the next five
slots. This yields 14,896 aligned rows. Source-row indices for current
observations are 85,099 through 99,994 (zero-based within the selected
100,000-row projection); the future target of the last window ends at 99,999.

The generator is
[`prepare_formal_prediction_bundle.py`](../../prediction_experiments/baseline_experiments/prepare_formal_prediction_bundle.py).
Its `--trace-only` mode can rebuild the CSVs from an authorized local trace and
a completed prediction benchmark. The formal FITS-Delta checkpoint is **not**
distributed, so these files, if permission is granted to publish them, permit
downstream update/search replay but do not by themselves reproduce model
training or regenerate its exact forecasts from scratch.

## File dictionary

All rows are aligned by position; there is no timestamp or node identifier in
these exports. Values use the trace's utilization percentage scale, not a
0--1 fraction. A forecast is a predicted percentage-scale value and need not
be interpreted as an additional physical measurement.

| File | Data rows | Bytes | Columns | Meaning |
| --- | ---: | ---: | --- | --- |
| `value.csv` | 14,896 | 157,056 | No header: column 1 = current CPU utilization; column 2 = current memory utilization | Last causally smoothed observation in each held-out input window |
| `cpu_predictions.csv` | 14,896 | 794,134 | `step_0` to `step_5` | `step_0` = current CPU observation; `step_1`--`step_5` = FITS-Delta CPU forecasts |
| `memory_predictions.csv` | 14,896 | 772,504 | `step_0` to `step_5` | `step_0` = current memory observation; `step_1`--`step_5` = FITS-Delta memory forecasts |

SHA-256 of the exact formal-run files:

```text
e2bf44333cd9c11cd15826f8a256c8928847e01ce72e82d900a54ddb281e845d  value.csv
64388fb14c0a53274d1ff31ae7444087ea137c42fa214dc09448beea3c5e06b9  cpu_predictions.csv
920a65d6416b8fdf604d15ae1dfd147014566034560af9d6df5940d2aaa52a04  memory_predictions.csv
```

After obtaining authorized copies, place **all three together** in one local
directory and run:

```sh
python -B scripts/verify_formal_prediction_inputs.py PATH_TO_THREE_CSVS
```

The checker verifies the frozen hashes, row counts, headers, finite numeric
values and `step_0`/`value.csv` alignment. It does not verify redistribution
rights. Do not put the CSVs into a public Git history until those rights have
been confirmed. If redistribution is not permitted, keep this metadata and
the generator, and direct readers to the original trace provider instead.
