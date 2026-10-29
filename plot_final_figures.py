"""Generate the three agreed ICC figures from a completed saved study.

Install: python -m pip install numpy matplotlib
Run: python plot_final_figures.py --study icc2027_all_failures_random3

Required inputs: study.sqlite and study_metadata.json inside --study.
Reads the database in read-only mode. No simulation or optimizer is executed.
Outputs three PNG images; no graphical display/backend is needed.

Latency: common successful failure scenarios across every method, Random seed,
and rate configuration. Scenarios are equally weighted; Random retains every
seed. Nonzero sensitivity levels average the link-layout means equally.
Timing: baseline completed attempts, including infeasible attempts; lines show
medians and shaded bands show the 25th to 75th percentiles.
Figure 2 still uses D15 data; only its in-plot device-count note is omitted.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import matplot2tikz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter
import numpy as np

SIZES = (5, 10, 15)
METHODS = ("MILP", "ComputationGreedy", "SRAMWorstFit", "FlashFFD", "Random")
METHOD_LABELS = ("MILP", "Computation Greedy", "SRAM Worst-Fit", "Flash FFD", "Random")
COLORS = dict(zip(METHODS, ("#193B63", "#18836C", "#A45C91", "#468FC5", "#D88700")))
MARKERS = dict(zip(METHODS, ("*", "o", "s", "^", "D")))
STYLES = dict(zip(METHODS, ("-", "--", "-.", ":", "-")))
LEVELS = (0.0, 0.25, 0.50, 0.75)


def read_results(study):
    study = Path(study).expanduser().resolve()
    db_path = study / "study.sqlite"
    meta_path = study / "study_metadata.json"
    if not db_path.is_file() or not meta_path.is_file():
        raise ValueError("--study must contain study.sqlite and study_metadata.json. "
                         "Extract the completed results ZIP first if necessary.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "finished" or meta.get("unresolved_trials", 0):
        raise ValueError("Use a finished study with no unresolved trials.")
    plan = meta["plan"]
    seeds = plan["random_seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Random seeds must be recorded and unique.")
    if not set(SIZES).issubset(plan["sizes"]):
        raise ValueError("The three figures require results for 5, 10 and 15 devices.")
    configs = {c["name"]: c for c in plan["configurations"]}
    if "baseline" not in configs or configs["baseline"]["level"] != 0:
        raise ValueError("Missing baseline rate configuration.")
    names_by_level = {level: [name for name, c in configs.items()
                             if c["level"] == level] for level in LEVELS}
    if any(not names for names in names_by_level.values()):
        raise ValueError("Expected sensitivity levels: 0%, 25%, 50%, 75%.")
    # Every expected method/seed identity must occur once per scenario/config.
    expected_ids = {(m, -1) for m in METHODS[:-1]} | {("Random", s) for s in seeds}
    expected_scenarios = {
        d: {s["id"] for s in plan["networks"][str(d)]["scenarios"]} for d in SIZES
    }
    seen = defaultdict(set)
    failure_counts = {}
    common = {d: set(scenarios) for d, scenarios in expected_scenarios.items()}
    timing_values = defaultdict(list)
    conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    try:
        rows = conn.execute("""SELECT network_size, configuration, scenario_id,
            method, seed, failure_count, feasible, completed_attempt,
            e2e_seconds, algorithm_ms FROM trials
            WHERE network_size IN (5, 10, 15)""").fetchall()
    finally:
        conn.close()
    for d, config, sid, method, seed, f, feasible, completed, latency, ms in rows:
        key = (d, config, sid)
        identity = (method, seed)
        if config not in configs or sid not in expected_scenarios[d] or identity not in expected_ids:
            raise ValueError("Unexpected trial identity: {} {}".format(key, identity))
        if identity in seen[key]:
            raise ValueError("Duplicate trial: {} {}".format(key, identity))
        seen[key].add(identity)
        if not completed:
            raise ValueError("Study contains an incomplete attempt: {}".format(key))
        if (d, sid) in failure_counts and failure_counts[d, sid] != f:
            raise ValueError("Inconsistent failure count for {}".format(key))
        failure_counts[d, sid] = f
        if feasible:
            if latency is None or not np.isfinite(latency) or latency < 0:
                raise ValueError("Invalid feasible latency: {}".format(key))
        else:
            common[d].discard(sid)
        if config == "baseline":
            if ms is None or not np.isfinite(ms) or ms < 0:
                raise ValueError("Invalid algorithm execution time: {}".format(key))
            timing_values[d, f, method].append(ms)
    for d in SIZES:
        for config in configs:
            for sid in expected_scenarios[d]:
                if seen.get((d, config, sid)) != expected_ids:
                    raise ValueError("Missing trials for D={}, {}, {}".format(d, config, sid))
        if not common[d]:
            raise ValueError("No common successful scenarios for D={}.".format(d))
    latency_values = defaultdict(list)
    for d, config, sid, method, seed, f, feasible, completed, latency, ms in rows:
        if sid in common[d]:
            latency_values[d, config, method].append(latency * 1000.0)
    config_means = {}
    for d in SIZES:
        for config in configs:
            for method in METHODS:
                values = latency_values[d, config, method]
                repetitions = len(seeds) if method == "Random" else 1
                if len(values) != len(common[d]) * repetitions:
                    raise ValueError("Unequal latency coverage for {}".format((d, config, method)))
                config_means[d, config, method] = float(np.mean(values))
    baseline = {(d, m): config_means[d, "baseline", m] for d in SIZES for m in METHODS}
    sensitivity = {
        (level, method): float(np.mean([config_means[15, name, method]
                                       for name in names_by_level[level]]))
        for level in LEVELS for method in METHODS
    }
    timings = {}
    for key, values in timing_values.items():
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        timings[key] = {"q1_ms": float(q1), "median_ms": float(median), "q3_ms": float(q3)}
    counts = {d: len(common[d]) for d in SIZES}
    return baseline, sensitivity, timings, counts


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 10,
        "axes.linewidth": 0.8, "savefig.facecolor": "white",
    })


def save(fig, path):
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path.resolve()


def plot_latency(means, out):
    colors = {5: "#D62728", 10: "#468FC5", 15: "#193B63"}
    markers = {5: "o", 10: "s", 15: "D"}
    labels = ("MILP", "Computation\nGreedy", "SRAM\nWorst-Fit", "Flash\nFFD", "Random")
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.86, bottom=0.13)
    for d in SIZES:
        values = [means[d, m] for m in METHODS]
        ax.plot(
            range(5), values,
            color=colors[d], marker=markers[d],
            linewidth=1.7, markersize=6, zorder=3,
            label=f"{d} devices",
        )
        ax.scatter([0], [values[0]], marker="*", s=150, color=colors[d],
                   edgecolor="white", linewidth=0.65, zorder=5)
        for index, value in enumerate(values):
            ax.annotate("{:,.2f}".format(value), (index, value), xytext=(0, 12),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=10.3, color=colors[d],
                        fontweight="bold" if index == 0 else "normal")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3, frameon=False,
        fontsize=12, columnspacing=3.5,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        fontsize=12,
        columnspacing=3.5,
    )
    ax.set_xticks(range(5), labels)
    ax.set_xlim(-0.4, 4.4)
    values = list(means.values())
    ax.set_ylim(min(1200, min(values) - 150), max(2850, max(values) + 150))
    ax.set_ylabel("Average end-to-end latency (ms)")
    ax.yaxis.set_major_locator(MultipleLocator(300))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color="#DDE2E8", linewidth=0.7)
    ax.set_axisbelow(True)
    matplot2tikz.save(
        out / "figure_1_average_latency_ms.tex",
        figure=fig,
    )
    return save(fig, out / "figure_1_average_latency_ms.png")


def method_handles(solid=False):
    return [Line2D([], [], color=COLORS[m], marker=MARKERS[m],
                   linestyle="-" if solid else STYLES[m], linewidth=1.7,
                   markersize=8 if m == "MILP" else 6, label=label)
            for m, label in zip(METHODS, METHOD_LABELS)]


def plot_sensitivity(values, out):
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.80, bottom=0.16)
    x = [100 * level for level in LEVELS]
    for method, label in zip(METHODS, METHOD_LABELS):
        y = [values[level, method] for level in LEVELS]
        ax.plot(
            x, y,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=STYLES[method],
            linewidth=2 if method == "MILP" else 1.7,
            markersize=9 if method == "MILP" else 6,
            markerfacecolor=(
                COLORS[method] if method in ("MILP", "Random") else "white"
            ),
            markeredgewidth=0.9,
            zorder=5 if method == "MILP" else 3,
            label=label,
        )
    for level in LEVELS:
        value = values[level, "MILP"]
        ax.annotate("{:,.2f}".format(value), (100 * level, value),
                    xytext=(0, -17), textcoords="offset points", ha="center", va="top",
                    fontsize=10.3, fontweight="bold", color=COLORS["MILP"])
    for level in LEVELS:
        value = values[level, "Random"]
        ax.annotate(
            "{:,.2f}".format(value),
            (100 * level, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.3,
            fontweight="bold",
            color=COLORS["Random"],
        )
    # This plot uses the 15-device results, without the in-plot device-count note.
    level = LEVELS[-1]
    lo, hi = values[level, "MILP"], values[level, "ComputationGreedy"]
    bx, tx = 78.0, 80.0
    ax.plot([75.7, bx], [lo, lo], color="#8996A4", linestyle=":", linewidth=1)
    ax.plot([75.7, bx], [hi, hi], color="#8996A4", linestyle=":", linewidth=1)
    ax.plot([bx, bx], [lo + 20, hi - 20], color="#495969", linewidth=1.2)
    ax.plot([bx - 1.7, bx + 1.7], [lo + 20, lo + 20], color="#495969", linewidth=1.2)
    ax.plot([bx - 1.7, bx + 1.7], [hi - 20, hi - 20], color="#495969", linewidth=1.2)
    ax.text(tx, (lo + hi) / 2, "Greedy vs MILP\n{:,.2f} ms".format(hi - lo),
            ha="left", va="center", fontsize=10, color="#495969", linespacing=1.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        fontsize=11,
        columnspacing=2.6,
        handlelength=2.4,
    )
    ax.set_xticks(x)
    ax.set_xlim(-4, 97)
    ax.set_ylim(min(2000, min(values.values()) - 300), max(4500, max(values.values()) + 200))
    ax.set_xlabel("Maximum link-rate reduction (%)", labelpad=10)
    ax.set_ylabel("Average end-to-end latency (ms)")
    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color="#DDE2E8", linewidth=0.7)
    ax.set_axisbelow(True)
    matplot2tikz.save(
    out / "link_sensitivity_ms.tex",
        figure=fig,
    )
    return save(fig, out / "figure_2_link_sensitivity_ms.png")


def plot_timing(stats, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.73,
                        bottom=0.22, wspace=0.40)
    fig.legend(handles=method_handles(solid=True), loc="center", bbox_to_anchor=(0.53, 0.94),
               ncol=5, frameon=False, fontsize=18, columnspacing=1.2, handlelength=1.6)
    for d, ax in zip(SIZES, axes):
        failures = sorted({f for size, f, method in stats if size == d})
        largest_q3 = 0
        for method in METHODS:
            values = [stats[d, f, method] for f in failures]
            largest_q3 = max(largest_q3, max(v["q3_ms"] for v in values))
            ax.fill_between(failures, [v["q1_ms"] for v in values], [v["q3_ms"] for v in values],
                            color=COLORS[method], alpha=0.10, linewidth=0)
            ax.plot(failures, [v["median_ms"] for v in values], color=COLORS[method],
                    marker=MARKERS[method], linestyle="-", linewidth=2 if method == "MILP" else 1.5,
                    markersize=8 if method == "MILP" else 5,
                    markerfacecolor=COLORS[method] if method in ("MILP", "Random") else "white",
                    markeredgewidth=0.9, zorder=5 if method == "MILP" else 3)
        ax.set_title("{} devices".format(d), fontsize=20, pad=14)
        ax.set_xticks(failures)
        ax.set_xlim(min(failures) - 0.25, max(failures) + 0.25)
        ax.set_ylim(0, largest_q3 * 1.12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel("Failed devices, F", fontsize=20, labelpad=10)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(axis="y", color="#DDE2E8", linewidth=0.7)
        ax.set_axisbelow(True)
        value = stats[d, failures[-1], "MILP"]["median_ms"]
        ax.annotate("{:.2f} ms".format(value), (failures[-1], value), xytext=(-4, 24 if d == 5 else 16),
                    textcoords="offset points", ha="right", color=COLORS["MILP"],
                    fontsize=20, fontweight="bold")
    axes[0].set_ylabel("Algorithm computation\ntime (ms)",fontsize=20, labelpad=12)
    return save(fig, out / "figure_3_algorithm_computation_time_ms.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", type=Path, required=True,
                        help="Completed results folder containing study.sqlite and study_metadata.json")
    parser.add_argument("--output-dir", type=Path, default=Path("final_figures"),
                        help="Destination for the three PNG figures (default: final_figures)")
    args = parser.parse_args()
    try:
        baseline, sensitivity, timings, counts = read_results(args.study)
        out = args.output_dir.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        setup_style()
        outputs = [plot_latency(baseline, out), plot_sensitivity(sensitivity, out), plot_timing(timings, out)]
    except (ValueError, KeyError, OSError, sqlite3.Error) as error:
        parser.exit(1, "Plot failed: {}\n".format(error))
    for d in SIZES:
        print("{} devices: {} common successful failure scenarios".format(d, counts[d]))
    print("Figure 2 uses 15-device data; its in-plot device-count note is removed.")
    print("Only saved results were read; no algorithms were rerun.")
    for output in outputs:
        print("Saved:", output)


if __name__ == "__main__":
    main()
