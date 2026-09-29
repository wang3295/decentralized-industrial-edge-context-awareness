# 100-node original-MPC-DP simulation: validated reconstruction

This runnable 12-file bundle combines eight byte-identical files from the
accepted 2026-09-14 source snapshot with four dependency files that were not
preserved in that snapshot. The added files are the current workspace copies:

| Added dependency | SHA-256 |
| --- | --- |
| `joint_experiments/run_cluster_and_gossip_budget_analysis.py` | `5f178a8ade395f6e3c2f24e1cec454b2818b3393fa01a0677093aca5b161fe8e` |
| `joint_experiments/run_threshold_aware_mpc_experiment.py` | `b086ea49e0fd6bd702f9dd43be7f85c38028517efe44fc60e0ea0ff9c02e3e9f` |
| `joint_experiments/run_network_scale_sensitivity.py` | `f50ffd1f32b3971c54dc30f84f6a4629189415409f90c00e2f4d04de14b2b646` |
| `edgebench_core/__init__.py` | `5a7d4d46c2e73b4999676a7a7ed28ba18112570d7e15738105ad449bde1c6617` |

The full reconstructed suite was run locally on 2026-09-29 with the three
formal-hash FITS-Delta CSVs, 25 independently generated networks, 120 requests
per network, 282 trace slots, four workers, and all four sections (`protocol`,
`robustness`, `trust`, `hyperparameter`). Validation reported `PASS`, 288,000
request rows, 2,400 network rows, 96 summary rows, a 50% update rate, and no
threshold-aware penalty. A keyed comparison against the accepted archive found
all 38 raw-table columns, 21 network-table columns, and 51 summary-table
columns exactly equal across all rows. The two runs used different operating
systems and Python/pandas versions, which changed CSV serialization bytes but
not parsed values. The comparison command is:

```sh
python -B scripts/compare_100node_rerun.py \
  PATH_TO_ACCEPTED_ARCHIVE PATH_TO_NEW_RERUN
```

To run from the package root with approved formal inputs available locally:

```sh
PYTHONPATH=formal_runs/100node_validated_reconstruction \
python -B formal_runs/100node_validated_reconstruction/search_experiments/run_industrial_100node_discovery_suite.py \
  --data-dir PATH_TO_FORMAL_INPUT_CSVS \
  --output-dir PATH_TO_NEW_OUTPUT_DIRECTORY \
  --repeats 25 --tasks-per-network 120 --trace-length 282 \
  --workers 4 --sections protocol,robustness,trust,hyperparameter
```

On Windows, set `PYTHONPATH` to the same directory using that shell's syntax.
Use a new output directory; do not overwrite the accepted archive. The archived
run used Linux/Python 3.12; this equivalence check used Windows/Python 3.10.
The source files alone are not enough to rerun the experiment: the three
formal-hash derived CSVs must be supplied or rebuilt through the documented
prediction pipeline. The data and checkpoints are not included here.

**Provenance limit:** matching all outputs establishes behavioural equivalence
of this reconstructed bundle under the formal inputs. It does not prove that
the accepted run loaded these exact four added dependency bytes, because their
hashes were not recorded at run time. Do not call the bundle a byte-identical
copy of all original dependencies.
