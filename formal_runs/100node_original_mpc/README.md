# Archived 100-node original-MPC-DP source snapshot

The eight Python files under `source_snapshot/` are byte-for-byte copies of the
files preserved with the archived formal run
`industrial_pc_100node_discovery_fits_delta_original_mpc_20260914`. Their
SHA-256 values were checked at copy time and are listed in `SHA256SUMS`. This folder keeps the reported
85.53% result tied to its actual source revision, which is not identical to
the later shared-core and scale code in this repository's main `raft/` tree.

**This snapshot is not a standalone runnable package.** The archived run did
not snapshot two indirectly imported joint-experiment helpers, its generated
input CSVs are not included here, and the archived shell launcher contained
machine-specific Linux/LXD paths. The launcher was deliberately not copied.
Do not combine these files with the current helper modules and claim exact
reproduction without first establishing their run-time versions and checking
the resulting output in a clean environment.

Recorded configuration: 100 nodes; 25 independently generated networks;
120 requests per network; FITS-Delta seed-2026 derived inputs; original MPC-DP
with `eta=1`, 50% update budget and no threshold penalty; 180 ms simulated
deadline. See `SHA256SUMS` here and `data/README.md` at the repository root
for source and input hashes. The accepted result CSVs and validation record
remain outside this public candidate.
