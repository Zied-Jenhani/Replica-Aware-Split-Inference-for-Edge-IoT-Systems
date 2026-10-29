# Referencing the repository in the paper

Once the scripts and datasets have been uploaded publicly, add the following
sentence at the end of the implementation/device-parameter subsection:

```latex
The simulation scripts and input datasets used in the evaluation
are publicly available in~\cite{jenhani_replica_code}.
```

Add this entry to the bibliography:

```bibtex
@misc{jenhani_replica_code,
  author       = {Zied Jenhani},
  title        = {{Replica-Aware Split Inference for Edge IoT Systems:
                   Simulation Scripts and Input Data}},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/Zied-Jenhani/Replica-Aware-Split-Inference-for-Edge-IoT-Systems}
}
```

Before submission, create a release for the exact code and inputs behind the
reported results, and record its tag or commit in the citation/README. No release
or archival DOI has been created by preparing this package. Software attribution
should be updated if other contributors should also be listed.

## Align the implementation description with the supplied code

The supplied `run_heuristics.py` selects the fastest compatible protocol directly
for each boundary after placing replicas. It does not solve a second MILP.
If these are the scripts used for the reported results, the baseline description
should say this. The direct selection is optimal for fixed hosts under the
independent-link model; computation-time comparisons must describe the
implementation actually timed.

The supplied sensitivity construction uses different reductions per link, bounded
by 25%, 50%, or 75%. Describe these percentages as maximum link-rate reductions,
rather than uniform global reductions.

The supplied `run_heuristics.py` does not contain LatencyGreedy. Include the
appropriate newer implementation if that baseline becomes part of the paper.

Publishing these files does not by itself establish that they are the exact
versions used for every existing figure. The completed study metadata and its
recorded algorithm/input hashes can establish that match.
