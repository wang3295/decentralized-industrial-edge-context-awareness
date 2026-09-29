# Code for decentralized adaptive context awareness in industrial edge computing

This repository makes the method and experiment **source code** associated with
the manuscript available for inspection. It includes FITS/FITS-Delta prediction,
the MPC-DP resource-state update policy and comparison policies, decentralized
candidate-discovery simulations, and generic physical-testbed node code. The
manuscript's final physical-testbed deployment scripts are **not** included.

This source-code package is not a complete one-command reproduction of every
paper result. Third-party traces, the three frozen downstream CSVs,
trained checkpoints, and physical-testbed raw records are not distributed here.
Measured component and physical-network latencies depend on the specified
hardware and environment. The 100-node discovery delays are simulated network
delays, not measurements from a physical 100-node network.

## What a reader can check now

| Manuscript component | Code | Availability boundary |
| --- | --- | --- |
| FITS/FITS-Delta and prediction baselines | `prediction_experiments/baseline_experiments/` and `scripts/run_prediction_benchmark.py` | Supply lawfully obtained, prepared Alibaba, GWA-T-12, and Google traces. No checkpoint is included. |
| MPC-DP recurrence and update baselines | `raft/edgebench_core/`, `raft/update_experiments/`, and `formal_runs/update_main_20260914/source_snapshot/` | A local full rerun with archived formal inputs matched 12 accepted update tables on every non-timing column. Hardware timing is not comparable; the input CSVs are not distributed. |
| Random-graph candidate discovery, robustness, sensitivity, and scale | `raft/search_experiments/`, `raft/joint_experiments/` and `formal_runs/100node_validated_reconstruction/` | Local reruns matched archived 100-node and 50/100/150-node parsed outputs with frozen inputs. Several historical dependency hashes were not recorded; the input CSVs are not included. |
| Physical testbed | `raft/testbed_deployment/` and `scripts/verify_physical20_summary.py` | Generic code and an archived-result checker are present. The final credential-free deployment bundle and raw results are not provided, so this directory cannot reproduce the reported physical latency. |

## Start here

Use a Python environment with the dependencies recorded in
`environments/requirements-formal-100node.txt` and
`raft/testbed_deployment/requirements-runtime.txt` as appropriate. These files
record experimental environments; a fresh, cross-platform installation has not
been certified. From the repository root, run:

```sh
python -B scripts/smoke_tests.py
python -B scripts/synthetic_integration_check.py
python -B scripts/publication_audit.py
```

The smoke test checks small model/protocol invariants and command-line imports.
The synthetic integration check runs a small deterministic pipeline through the
100-node code using artificial inputs; its outputs are **not** paper results.
Neither check reproduces a manuscript table. The publication audit checks this
tree for accidentally included private artifacts and common credential/path
patterns; it does not validate scientific results.

For the nine-model benchmark with prepared trace files, see
`scripts/run_prediction_benchmark.py --help`. For the reconstructed 100-node
suite, use the command and provenance notes in
`formal_runs/100node_validated_reconstruction/README.md`. Data sources,
transformations and SHA-256 fingerprints are documented in `data/README.md`.
`MANUSCRIPT_MAP.md` maps the revised paper's tables and figures to source files
and states the verification boundary for each.
`REPRODUCE.md` gives runnable commands, expected checks, and the remaining
verification work.
The named original traces must be obtained from their own providers; do not
substitute other local CSVs for the frozen formal inputs and call the resulting
numbers a reproduction of the paper.

## Verification status and interpretation

The input-free smoke and synthetic integration checks passed locally. The
archived update comparison, complete reconstructed 100-node suite and scale simulation were rerun
locally against archived frozen inputs: parsed raw, per-network and summary
outputs matched their accepted archives on comparable fields. This is
**output-equivalence evidence**, not proof that all historical dependency bytes
were preserved. Prediction, update, scale and physical-testbed results have not
all been independently regenerated from publicly obtainable inputs in a clean
environment. Please interpret the available source and the manuscript claims
within those limits.

## Availability and rights

The authors have chosen **not to attach a software licence at this time**.
Public visibility alone does not grant general permission to use, modify or
redistribute the code. If published, it should therefore be described as
*publicly available source code*, not as *open-source software*. Rights to the third-party traces remain with their
respective providers. Trained weights are intentionally not included.

For citation, see `CITATION.cff`; the manuscript's final bibliographic details
will be updated after publication.
