# Replica-Aware Split Inference for Edge IoT Systems

Python simulation scripts and input datasets for latency-aware replica placement
after device failures in distributed TinyML inference. The MILP uses SciPy's
HiGHS interface. Evaluated network sizes are 5, 10, and 15 devices.

This source-and-input package supports rerunning the experiments. Previously
computed study databases, summary tables, and paper figures are not bundled.

## Contents

| Path | Purpose |
| --- | --- |
| `solve_milp.py` | Solve and independently verify a selected failure scenario |
| `run_heuristics.py` | Random, ComputationGreedy, SRAMWorstFit, and FlashFFD |
| `run_link_rate_study.py` | Enumerate failures, run algorithms, and checkpoint results |
| `plot_link_rate_study.py` | Build summaries and diagnostic figures |
| `plot_final_figures.py` | Read a completed study and produce the three paper-style figures |
| `inputs_D5/`, `inputs_D10/`, `inputs_D15/` | Baseline inputs in the runner's expected directory layout |
| `generated_inputs/` | All 30 supplied configurations, including baseline and sensitivity inputs |
| `INPUTS.md` | Input definitions and sensitivity construction |
| `PAPER_REFERENCE.md` | Suggested LaTeX citation and BibTeX entry |
| `GITHUB_UPLOAD.md` | Initial upload commands |
| `VALIDATION.md` | Checks performed when preparing this package |

The five Python scripts retain the supplied source content. The three
`inputs_D*` folders are byte-identical copies of the corresponding
`generated_inputs/D*/baseline` folders. Always keep the four algorithm/study
scripts together so their local imports resolve.

## Installation

Use Python 3.10 or later. From the repository root on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`matplot2tikz` is required by the supplied final-figure script for its TeX exports.
SQLite is provided by Python's standard library. See `VALIDATION.md` for the
environment used for package checks; this is not a record of the original
experiment workstation's environment.

## Start with one verified MILP scenario

```bash
python solve_milp.py --inputs inputs_D5 --failed 3,4 --protocol joint --verify --output results/example_D5
```

`--verify` additionally compares the small scenario against exhaustive host
enumeration. The output includes `solution.json`, placements, transfers,
resource usage, and an audit of the optimization model.

## Run the four baselines for the 5-device network

```bash
python run_heuristics.py --inputs inputs_D5 --protocol joint --random-runs 3 --output results/heuristics_D5
```

This evaluates all 15 eligible failure combinations. Repeat with `inputs_D10`
or `inputs_D15` and a distinct output directory for the other network sizes.
`summary.csv` and `run_metadata.json` are generated. The script prints a legacy
suggestion to run `calculate_metrics.py`; that companion is not included here.
Use the integrated study below for the bundled summary/figure workflow.

## Check the integrated workflow using a small sample

```bash
python run_link_rate_study.py --inputs-root . --sizes 5 10 15 --levels 0.25 0.5 0.75 --link-seeds 2026 --random-runs 3 --max-per-failure-count 1 --output results/workflow_check --no-plots
```

This is a sampled workflow check, not the full paper evaluation. It runs 392
algorithm trials across 14 selected failure scenarios and four configurations.

## Run the full study

```bash
python run_link_rate_study.py --inputs-root . --sizes 5 10 15 --levels 0.25 0.5 0.75 --link-seeds 2026 2027 2028 --random-runs 3 --output results/full_study
```

Failures are independent combinations, with the source device excluded:

| Devices | Failure counts | Eligible combinations |
| --- | --- | --- |
| 5 | 1 through 4 | 15 |
| 10 | 1 through 5 | 381 |
| 15 | 1 through 5 | 3,472 |

The baseline plus nine sensitivity configurations, five methods, and three
Random repetitions give 270,760 algorithm trials. This command performs a new
study. Its timings depend on the execution environment. Both runner defaults
use 30 Random repetitions; the commands above explicitly request three.

To resume, repeat the same command with `--resume`. Keep the paths, arguments,
scripts, input files, and Python environment unchanged. To rebuild summaries:

```bash
python run_link_rate_study.py --output results/full_study --summarize-only
```

The runner regenerates configurations from the baseline copies using its
deterministic link-rate construction. The supplied `generated_inputs/` directory
also preserves the exact uploaded configurations for inspection or individual
runs through `solve_milp.py` and `run_heuristics.py`.

## Generate the three final figures

After the full study finishes with no unresolved trials:

```bash
python plot_final_figures.py --study results/full_study --output-dir results/final_figures
```

The script reads `study.sqlite` and `study_metadata.json`. It creates PNG and PDF
versions of all three figures, plus TeX exports for the first two. A compact
results ZIP without the database cannot be used with this script.

Latency figures use common successful scenarios across all methods, Random
seeds, and configurations. The sensitivity figure uses the 15-device network.
Timing uses completed baseline attempts, including infeasible attempts, with
medians and interquartile bands. The supplied first figure places methods on
the horizontal axis and device counts in the legend.

## Model and implementation scope

- Surviving primary segments stay fixed; failed segments are replicated on survivors.
- MILP jointly optimizes placement and compatible link protocols.
- The heuristics select the fastest compatible protocol directly after placement.
  They do not invoke another MILP. For this independent-link model, this minimizes
  transfer latency for fixed hosts. It does not imply globally optimal placement.
- Co-located consecutive segments have zero transfer latency.
- Inputs use the `esp32_psram` policy, including the explicit ESP32 model quota
  of 1,363,148 bytes and per-segment overhead. Runtime requirements are modeled.
- Sensitivity levels bound per-link reductions; they are not uniform reductions
  applied to every link. See `INPUTS.md`.
- Computation parameters and link rates are model inputs; these scripts do not
  measure MCU inference times or radio channels.
- OTA recovery duration, detection delay, energy optimization, and packet loss
  are outside these scripts' optimization objective.
- The supplied script contains four heuristic baselines. LatencyGreedy is not
  present in this uploaded version.

## Citation

See `CITATION.cff` and `PAPER_REFERENCE.md`. When preparing a publication release,
record the exact Git commit or release tag used for the reported results.
