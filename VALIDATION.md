# Package checks — 29 October 2026

These checks verify the uploaded source/input package and a small rerun. They
do not reproduce or certify all results reported in the paper.

## Environment used for these checks

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |
| matplot2tikz | 0.5.4 |
| webcolors | 25.10.0 |

This is the package-check environment. The original experiment environment and
completed study metadata were not supplied in this upload. Exact original
package versions and workstation timings should be recorded with the
publication release rather than inferred from these checks.

## Completed checks

1. All five Python scripts and all 120 supplied CSV/JSON input files were compared
   byte-for-byte with the uploaded sources. All match.
2. The 12 files under the three `inputs_D*` convenience folders match the supplied
   baseline configurations byte-for-byte.
3. All 30 input configurations pass `load_inputs()` validation. Every original
   primary deployment passes the memory and link feasibility checks.
4. Failure enumeration produces 15, 381, and 3,472 combinations for 5, 10, and 15
   devices, respectively.
5. All 27 nonbaseline configurations match `vary_rates()` exactly when generated
   from the included baseline data and the recorded level/seed.
6. The 5-device failure set `{3,4}` solves to optimality and matches exhaustive
   enumeration of nine host assignments. Independent resource/link/objective
   checks pass. In this rerun, the optimum is 1.63477103958841 seconds.
7. The standalone heuristic command completes all 15 D5 scenarios using three
   Random seeds: 45 Random trials and 15 trials for each deterministic method.
8. An integrated sampled study completes 392/392 trials with zero unresolved
   attempts. It uses sizes 5/10/15, one selected combination per failure count,
   four rate configurations, one link seed, and three Random seeds. The
   checkpointed database was reopened and its row count checked.
9. The final-figure script reads that sampled database and successfully writes
   three PNGs, three PDFs, and two TeX exports. These are workflow-check figures,
   not replacements for the paper's full-study figures.

The sample outputs and temporary dependencies are deliberately outside the
release package. `SHA256SUMS` records the five scripts and the 132 packaged
input files, including the convenience copies. Verify them from the repository
root with:

```bash
sha256sum -c SHA256SUMS
```

## Before assigning a publication release

Match the completed study's recorded script/input hashes and settings to the
committed files. Confirm the number of Random seeds, scope of failure
enumeration, protocol-selection implementation, and link-rate construction.
The uploaded heuristic version does not include LatencyGreedy. Previously
computed tables, original databases, and original paper figures are not
included in this source/input package.
