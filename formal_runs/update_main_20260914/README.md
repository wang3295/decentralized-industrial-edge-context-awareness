# Archived source for the industrial-PC state-update comparison

The two files in `source_snapshot/` are byte-for-byte copies of the Python
sources retained with the formal `fits_delta_unified_20260914/industrial_pc`
run. They are kept separately because the current shared core is a later
revision. The main manuscript reports a 50% update-budget WRMSE of 0.422 from
this experiment.

| File | SHA-256 |
| --- | --- |
| `source_snapshot/update_policies.py` | `3fd1a588b166565bfbe38e46e4a5ccb8525d3d5feb8743bca39a656992d2bf9e` |
| `source_snapshot/run_update_experiments.py` | `bc7e3e8a4b0dde5796c8bdbe2b4482225c2b9426cde2570419b843ae9771dcc4` |

The exact derived inputs and final output tables are **not** included here.
`scripts/run_archived_update.py` supplies the historical package import layout
without editing either archived file. A local full rerun using the frozen
formal inputs matched every non-timing column in all 12 accepted CSV tables;
`scripts/compare_update_rerun.py` performs that keyed comparison. CPU decision
latencies are specific to the machine and are excluded. This is conditional
output-equivalence evidence, not a standalone public-input reproduction.
Do not replace the archived files with same-named files from the current
`raft/` tree while claiming to rerun the original experiment.
