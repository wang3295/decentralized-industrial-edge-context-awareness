# Reproduction and verification guide

This guide separates checks that work without private inputs from conditional reruns and
results that cannot yet be independently reproduced from this repository.
Run commands from the repository root. Use a fresh output directory for every
formal rerun, retain its `run_metadata.json` and `validation.json`, and record
the exact source commit and input hashes. Do not substitute same-named CSVs
unless their SHA-256 values match `data/README.md`.

## 1. Requirements and input-free checks

The archived 100-node run used Linux, Python 3.12.14 and the package versions
listed in `environments/requirements-formal-100node.txt` (including the CPU
PyTorch wheel). These pins are a transcribed environment record, **not yet a
fresh-install-validated lockfile**. The following checks were run locally with
Python 3.10; a fresh Linux/Python 3.12 installation remains an independent
verification item.

```sh
python -B scripts/smoke_tests.py
python -B scripts/synthetic_integration_check.py
python -B scripts/publication_audit.py
```

The first command checks unit-level invariants and CLI imports. The second
generates temporary **synthetic**, constant-horizon predictions and runs a
small protocol/hyperparameter pipeline through the 100-node source. Its
expected final line is `Synthetic integration check passed; no manuscript
result was reproduced.` The third scans for common private-data, secret and
path patterns; it is not a scientific or legal audit. None of these commands
recreates a manuscript table.

## 2. Prediction benchmark (Tables II and III)

The FITS and FITS-Delta implementations and the unified nine-model comparison
are in `prediction_experiments/baseline_experiments/`. Obtain the named traces
from their providers and prepare the input CSVs as described in
`data/README.md` before running:

```sh
python -B scripts/run_prediction_benchmark.py \
  --alibaba-csv PATH_TO_PREPARED_ALIBABA_CSV \
  --gwa-csv PATH_TO_PREPARED_GWA_CSV \
  --google-csv PATH_TO_PREPARED_GOOGLE_CSV \
  -- --datasets Alibaba GWA-T-12 Google --output PATH_TO_NEW_OUTPUT_DIRECTORY
```

Consult `scripts/run_prediction_benchmark.py --help` and the runner's `--help`
for optional arguments. This wrapper changes data locations only. The exact
upstream-to-prepared provenance for GWA-T-12 and Google is not fully closed;
the complete nine-model benchmark has not been freshly rerun from verified
official downloads in an isolated environment. Consequently, the command is
a route to run the code, **not a promise of identical paper scores**.

## 3. MPC-DP update comparison (Table IV)

`formal_runs/update_main_20260914/source_snapshot/` preserves the two
recorded source files for the reported update run. The wrapper below binds
the archived core to the import name expected by the **unchanged** archived
runner. The executable recurrence and policy-level tests are also available
under `raft/edgebench_core/` and `raft/update_experiments/`. The exact three
formal derived inputs are not distributed; their schemas and SHA-256
fingerprints are in `data/README.md`. With an authorized copy, verify them
first, then run the full comparison:

```sh
python -B scripts/verify_formal_prediction_inputs.py PATH_TO_FORMAL_INPUT_DIRECTORY
python -B scripts/run_archived_update.py -- \
  --data-dir PATH_TO_FORMAL_INPUT_DIRECTORY \
  --output-dir PATH_TO_NEW_UPDATE_OUTPUT_DIRECTORY \
  --validation-ratio 0.2 --random-repeats 30 --horizon 6
```

Do **not** add `--main-only` when comparing with the accepted full archive:
that speed path runs Random only once, whereas the formal comparison uses 30
Random repetitions. With access to the accepted result directory, compare:

```sh
python -B scripts/compare_update_rerun.py \
  PATH_TO_ACCEPTED_UPDATE_RESULT_DIRECTORY PATH_TO_NEW_UPDATE_OUTPUT_DIRECTORY
```

A local full rerun on 2026-09-29 with formal-hash inputs matched every
non-timing column in **all 12 accepted CSV tables** (including 170 raw-run
rows, 25 budget-summary rows, five B=0.5 main-table rows and 600 update-loss
rows). For B=0.5, MPC-DP WRMSE was 0.4219177657. Decision-time columns are
deliberately excluded because the original industrial PC and the local
verification machine differ. This remains a conditional rerun, not a
clean-environment public-input reproduction. Do not replace the archived
source with newer `raft/` files and call it the historical run.

## 4. Candidate-discovery simulation (100-node figures)

The runnable 12-file reconstruction is in
`formal_runs/100node_validated_reconstruction/`; read its README for the four
indirect-dependency provenance limit. Given the three **exact formal-hash**
input files, the full command is:

```sh
python -B scripts/verify_formal_prediction_inputs.py PATH_TO_FORMAL_INPUT_DIRECTORY
python -B formal_runs/100node_validated_reconstruction/search_experiments/run_industrial_100node_discovery_suite.py \
  --data-dir PATH_TO_FORMAL_INPUT_DIRECTORY \
  --output-dir PATH_TO_NEW_OUTPUT_DIRECTORY \
  --repeats 25 --tasks-per-network 120 --trace-length 282 \
  --workers 4 --sections protocol,robustness,trust,hyperparameter
```

`validation.json` should report `status: PASS`, 288,000 raw request rows,
2,400 per-network rows and 96 summary groups. The network (25 generated
instances per condition), not an individual request, is the independent
statistical unit. A local rerun with archived inputs matched every parsed
raw, network and summary value in the accepted archive; CSV byte hashes
differed across operating systems/versions. The code used here is therefore
**output-equivalent under those inputs**, not proven byte-identical to all
historical dependencies. The 100-node delays are modeled, not physical-network
latencies. Neither the formal inputs nor the accepted output archive is
distributed in this repository.

If the accepted result directory is available privately, the read-only
accounting check is:

```sh
python -B scripts/verify_100node_nominal.py PATH_TO_ACCEPTED_100NODE_RESULT_DIRECTORY
```

This reaggregates archived nominal requests; it does not run a simulation.
After a new full run, `scripts/compare_100node_rerun.py` compares its parsed
outputs against the accepted archive. Again, that archive is not included.

## 5. Scalability simulation (50/100/150-node figure)

With the same three formal-hash inputs, the five-configuration scale run is:

```sh
python -B raft/joint_experiments/run_deployment_equivalent_network_scale.py \
  --data-dir PATH_TO_FORMAL_INPUT_DIRECTORY \
  --output-dir PATH_TO_NEW_SCALE_OUTPUT_DIRECTORY \
  --node-counts 50,100,150 --repeats 25 --tasks-per-network 200 \
  --workers 4 --trace-length 228 --trace-start 94 --offset-step 1 \
  --budget 0.5 --eta 1.0 --delta-threshold 5.0 \
  --lyapunov-v0 0.001 --lyapunov-power 0.5 --method-set paper-five
```

A local rerun on 2026-09-29 with the archived formal inputs passed its
validation and matched all **75,000 raw, 375 per-network and 15 summary rows**
of the accepted scale archive after keyed CSV parsing. The full-workflow
discovery rates were 93.12%, 91.92% and 89.68% at 50, 100 and 150 nodes.
The historical metadata records the shared update-core hash, but not all
imported source hashes; this is output-equivalence evidence, not complete
byte-level historical source provenance. The comparison command, when the
accepted archive is available, is:

```sh
python -B scripts/compare_scale_rerun.py \
  PATH_TO_ACCEPTED_SCALE_RESULT_DIRECTORY PATH_TO_NEW_SCALE_OUTPUT_DIRECTORY
```

The 25 networks at each scale are the independent units; the 200 requests
within each network are repeated observations. Delays are simulated, not
measured from a 150-node physical deployment.

## 6. Physical testbed (Table VI)

`raft/testbed_deployment/` contains generic node and policy code, **not the
exact final deployment launcher**. The final run used three physical hosts,
20 virtualized ECNs, five matched blocks and 100 formal requests per
method/block (2,500 formal requests total). Its preceding 100-request
instrument check was excluded. With the authors' private accepted result
directory, the following read-only verifier checks rates, successful-request
P95 origin-side decision latency, message accounting and audit records:

```sh
python -B scripts/verify_physical20_summary.py PATH_TO_ACCEPTED_20NODE_RESULT_DIRECTORY
```

The directory and trained checkpoint are not distributed. The checker does
not redeploy nodes or reproduce hardware-/Wi-Fi-dependent latency. The final
launcher still needs a credential-free, parameterized public version, with
its differences from the archived exact source disclosed. Do not describe
the generic code as the source that generated Table VI.

## 7. Current limitations and remaining verification work

- Verify official download/version-to-processed-file lineage and reuse terms
  for Alibaba, GWA-T-12 and Google. Do not mirror third-party traces without
  permission.
- The three derived formal CSVs and trained weight are not distributed. Validate
  a lawful, fully documented reconstruction from available inputs without them.
- Reconcile every remaining final table/figure with its exact code,
  configuration, inputs, raw outputs and plotting command, especially disk-I/O
  results. The update, 100-node and scale reruns have passed local
  output-equivalence checks, but lack a public-input clean-environment trial.
- Test dependency installation and representative full reruns in a clean
  environment, including an external-reader trial.
- Prepare and test a sanitized physical-testbed deployment path; audit raw
  records before any sharing.
- Confirm `CITATION.cff` author metadata. The authors currently choose no
  software licence; public visibility would not grant general reuse rights.
- For the manuscript revision, record a fixed repository version, verify
  reviewer access, and cite the exact repository URL and version in the
  manuscript and response letter.

Until the verification and publication gaps above are closed, describe this
artifact as **source code with partial local verification**, not as a fully
reproducible repository. Its public availability, if any, must be verified from
the exact repository URL and version cited in the manuscript and response letter.
