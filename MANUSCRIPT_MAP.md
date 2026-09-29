# Manuscript result to code map

This map follows the author's revised manuscript source checked on 2026-09-28.
It identifies implementations, not a guarantee that every table and figure can
be regenerated from this repository alone. Check labels again if the manuscript
changes.

| Manuscript item | Code in this repository | Verification boundary |
| --- | --- | --- |
| Prediction comparison, Tables II–III and prediction figure | `scripts/run_prediction_benchmark.py`; `prediction_experiments/baseline_experiments/run_all_models_unified.py`, `run_baselines.py`, `benchmark_models_on_vm.py`, `plot_fits_delta_prediction_figure.py` | FITS baseline and FITS-Delta code are present. Public traces must be obtained and prepared separately; the full published scores have not been clean-rerun from those sources here. |
| Main update comparison and communication-budget sweep, Table IV and update figures | `formal_runs/update_main_20260914/source_snapshot/`; `scripts/run_archived_update.py`; `scripts/compare_update_rerun.py` | A full local rerun with formal-hash inputs matched all non-timing columns in 12 accepted update tables. CPU decision latencies are hardware-dependent; the input CSVs are absent from this candidate. |
| MPC-DP parameter sensitivity and disk-I/O extension | `raft/joint_experiments/run_weight_eta_task_sensitivity.py`; the disk-I/O prediction and update scripts in `prediction_experiments/baseline_experiments/` and `raft/update_experiments/` | Relevant code is present; exact final figure-source reconciliation remains pending. |
| 100-node candidate-discovery comparison, robustness, ablations and sensitivity | `formal_runs/100node_original_mpc/source_snapshot/`; runnable `formal_runs/100node_validated_reconstruction/`; `scripts/compare_100node_rerun.py` | A local full rerun with archived inputs matched 288,000 raw, 2,400 per-network, and 96 summary rows. Four indirect historical source hashes were not recorded, and the inputs are not distributed. |
| Physical testbed, Table VI | Generic `raft/testbed_deployment/`; `scripts/verify_physical20_summary.py` | The generic code is **not** the exact final deployment runner. The checker needs the authors' archived result directory, which is not included. |
| 50/100/150-node scalability figure | `raft/joint_experiments/run_deployment_equivalent_network_scale.py`; `scripts/compare_scale_rerun.py` | A local rerun with formal-hash inputs matched all 75,000 raw, 375 per-network and 15 summary rows in the accepted scale archive. The historical metadata records the shared-core hash but not every imported source hash; the inputs and archive are not distributed. |

The 100-node delays are modeled simulation delays. The physical-testbed P95 is
origin-side wall-clock decision latency among successful requests. These are
different quantities and should not be compared as though measured by the same
protocol.
