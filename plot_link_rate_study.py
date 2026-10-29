"""Summarize and plot link-rate studies without running any algorithms.

Replot saved compact data: python plot_link_rate_study.py --output link_rate_study
Rebuild compact data from the database: python run_link_rate_study.py
    --output link_rate_study --summarize-only
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np

METHODS = ('MILP', 'Random', 'ComputationGreedy', 'SRAMWorstFit', 'FlashFFD')
LABELS = ('MILP', 'Random', 'Comp. Greedy', 'SRAM Worst-Fit', 'Flash FFD')
SHORT = ('MILP', 'Random', 'Comp.\nGreedy', 'SRAM\nWorst-Fit', 'Flash\nFFD')
COLORS = ('#193B63', '#D88700', '#18836C', '#A45C91', '#468FC5')
MARKERS = ('o', 's', '^', 'D', 'v')
POOLS = ('model', 'flash', 'sram', 'psram')
POOL_LABELS = ('Model storage quota', 'Complete OTA-slot image', 'SRAM', 'PSRAM')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def distribution(values):
    if not len(values):
        return dict(n=0, mean=None, med=None, q1=None, q3=None, whislo=None, whishi=None, fliers=[])
    values = np.asarray(values, dtype=float)
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    inside = values[(values >= low) & (values <= high)]
    return dict(n=len(values), mean=float(values.mean()), med=float(med), q1=float(q1), q3=float(q3),
        whislo=float(inside.min()), whishi=float(inside.max()),
        fliers=np.unique(values[(values < low) | (values > high)]).tolist())


def bucket():
    return dict(trials=0, feasible=0, completed=0, unresolved=0, reference_optimal=0,
        recovered_reference_optimal=0, avoidable_failures=0, reference_infeasible=0,
        seed_counts=defaultdict(lambda: [0, 0]), statuses=defaultdict(int),
        latency=array('d'), computation=array('d'), communication=array('d'),
        timing=array('d'), gap=array('d'), paired_method=array('d'), paired_milp=array('d'),
        resources={pool: array('d') for pool in POOLS})


def group_statistics(group):
    rates = [100 * good / total for good, total in group['seed_counts'].values()]
    return dict(trial_count=group['trials'], feasible_trials=group['feasible'],
        completed_attempts=group['completed'], unfinished_attempts=group['unresolved'],
        feasibility_percent=100 * group['feasible'] / group['trials'] if group['trials'] else None,
        random_seed_rate_sd=statistics.stdev(rates) if len(rates) > 1 else None,
        reference_proven_infeasible_trials=group['reference_infeasible'],
        reference_optimal_trials=group['reference_optimal'],
        failures_on_reference_optimal_trials=group['avoidable_failures'],
        success_on_reference_optimal_percent=100 * group['recovered_reference_optimal'] /
            group['reference_optimal'] if group['reference_optimal'] else None,
        status_counts=dict(group['statuses']), latency=distribution(group['latency']),
        computation=distribution(group['computation']), communication=distribution(group['communication']),
        computation_time_ms=distribution(group['timing']), paired_gap_percent=distribution(group['gap']),
        paired_method_latency=distribution(group['paired_method']),
        paired_milp_latency=distribution(group['paired_milp']),
        resources={pool: distribution(group['resources'][pool]) for pool in POOLS})


def merge_buckets(groups):
    combined = bucket()
    for group in groups:
        for field in ('trials', 'feasible', 'completed', 'unresolved', 'reference_optimal',
                      'recovered_reference_optimal', 'avoidable_failures', 'reference_infeasible'):
            combined[field] += group[field]
        for field in ('latency', 'computation', 'communication', 'timing', 'gap', 'paired_method', 'paired_milp'):
            combined[field].extend(group[field])
        for pool in POOLS:
            combined['resources'][pool].extend(group['resources'][pool])
        for seed, counts in group['seed_counts'].items():
            combined['seed_counts'][seed][0] += counts[0]
            combined['seed_counts'][seed][1] += counts[1]
        for status, count in group['statuses'].items():
            combined['statuses'][status] += count
    return combined


def flat_row(size, configuration, failure, method, stat, scenario_count):
    row = dict(network_size=size, configuration=configuration['name'], level=configuration['level'],
        link_seed=configuration['link_seed'], failure_count=failure, method=method,
        scenario_count=scenario_count, trial_count=stat['trial_count'], feasible_trials=stat['feasible_trials'],
        feasibility_percent=stat['feasibility_percent'], random_seed_rate_sd=stat['random_seed_rate_sd'],
        completed_attempts=stat['completed_attempts'], unfinished_attempts=stat['unfinished_attempts'],
        reference_proven_infeasible_trials=stat['reference_proven_infeasible_trials'],
        reference_optimal_trials=stat['reference_optimal_trials'],
        failures_on_reference_optimal_trials=stat['failures_on_reference_optimal_trials'],
        success_on_reference_optimal_percent=stat['success_on_reference_optimal_percent'],
        mean_e2e_seconds_own_feasible=stat['latency']['mean'],
        median_e2e_seconds_own_feasible=stat['latency']['med'],
        mean_computation_seconds_own_feasible=stat['computation']['mean'],
        mean_communication_seconds_own_feasible=stat['communication']['mean'],
        median_algorithm_ms_completed=stat['computation_time_ms']['med'],
        q1_algorithm_ms_completed=stat['computation_time_ms']['q1'],
        q3_algorithm_ms_completed=stat['computation_time_ms']['q3'],
        paired_trial_count=stat['paired_gap_percent']['n'],
        paired_method_mean_e2e_seconds=stat['paired_method_latency']['mean'],
        paired_milp_mean_e2e_seconds=stat['paired_milp_latency']['mean'],
        mean_paired_gap_percent=stat['paired_gap_percent']['mean'])
    for pool in POOLS:
        row[f'mean_{pool}_utilization_percent'] = stat['resources'][pool]['mean']
    return row


def across_seeds(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['network_size'], row['level'], row['failure_count'], row['method']].append(row)
    result = []
    for (size, level, failure, method), sample in groups.items():
        row = dict(network_size=size, level=level, failure_count=failure, method=method,
                   layout_count=len(sample), link_seeds=';'.join(str(r['link_seed']) for r in sample if r['link_seed'] is not None),
                   unfinished_attempts=sum(r['unfinished_attempts'] for r in sample))
        for field in ('feasibility_percent', 'mean_paired_gap_percent', 'mean_e2e_seconds_own_feasible',
                      'median_algorithm_ms_completed'):
            values = [r[field] for r in sample if r[field] is not None]
            row[f'{field}_layout_mean'] = statistics.mean(values) if values else None
            row[f'{field}_layout_min'] = min(values) if values else None
            row[f'{field}_layout_max'] = max(values) if values else None
            row[f'{field}_layouts_with_values'] = len(values)
        result.append(row)
    return result


def build_summaries(database, output, metadata):
    plan = metadata['plan']
    by_name = {c['name']: c for c in plan['configurations']}
    datasets, summaries, failure_summaries, device_summaries, unresolved = [], [], [], [], []
    for prepared in metadata['prepared_inputs']:
        size, name = prepared['network_size'], prepared['configuration']
        configuration = by_name[name]
        scenarios = plan['networks'][str(size)]['scenarios']
        scenario_counts = defaultdict(int)
        for scenario in scenarios:
            scenario_counts[len(scenario['failed'])] += 1
        # Capacities come from validated effective device values stored by the loader.
        # The metadata's fingerprint already includes the ESP32 model quota override.
        device_csv = output / prepared['directory'] / 'devices.csv'
        case = json.loads((output / prepared['directory'] / 'case.json').read_text())
        limits = {}
        with device_csv.open(newline='', encoding='utf-8-sig') as stream:
            for node in csv.DictReader(stream):
                model = int(node.get('M_model_budget_bytes') or max(0, int(node['M_OTA_bytes']) - int(node['M_base_bytes'])))
                if node['device_type'].upper().startswith('ESP32'):
                    model = case['esp32_model_budget_bytes']
                limits[int(node['device_id'])] = [model, int(node['M_OTA_bytes']), int(node['M_SRAM_bytes']), int(node['M_PSRAM_bytes'])]
        groups = {(failure, method): bucket() for failure in sorted(scenario_counts) for method in METHODS}
        device_totals = defaultdict(lambda: [0, 0.0, 0.0])
        query = '''SELECT t.*, r.status AS reference_status, r.e2e_seconds AS reference_e2e
            FROM trials AS t LEFT JOIN trials AS r ON
            r.network_size=t.network_size AND r.configuration=t.configuration AND
            r.scenario_id=t.scenario_id AND r.method='MILP' AND r.seed=-1
            WHERE t.network_size=? AND t.configuration=?'''
        for row in database.execute(query, (size, name)):
            failure, method = row['failure_count'], row['method']
            group = groups[failure, method]
            group['trials'] += 1
            group['feasible'] += row['feasible']
            group['completed'] += row['completed_attempt']
            group['unresolved'] += 1 - row['completed_attempt']
            group['statuses'][row['status']] += 1
            if method == 'Random':
                group['seed_counts'][row['seed']][0] += row['feasible']
                group['seed_counts'][row['seed']][1] += 1
            if row['completed_attempt']:
                group['timing'].append(row['algorithm_ms'])
            else:
                unresolved.append(dict(network_size=size, configuration=name, scenario_id=row['scenario_id'],
                    method=method, seed=row['seed'], status=row['status'], feasible=bool(row['feasible']),
                    e2e_seconds=row['e2e_seconds'], objective_bound_seconds=row['objective_bound_seconds'],
                    solver_relative_gap=row['solver_relative_gap'], algorithm_ms=row['algorithm_ms']))
            if row['reference_status'] == 'optimal':
                group['reference_optimal'] += 1
                group['recovered_reference_optimal'] += row['feasible']
                group['avoidable_failures'] += 1 - row['feasible']
            elif row['reference_status'] == 'proven_infeasible':
                group['reference_infeasible'] += 1
            if not row['feasible']:
                continue
            for field, column in (('latency', 'e2e_seconds'), ('computation', 'computation_seconds'),
                                  ('communication', 'communication_seconds')):
                group[field].append(row[column])
            if row['reference_status'] == 'optimal':
                ref = row['reference_e2e']
                gap = 100 * (row['e2e_seconds'] - ref) / ref if ref > 0 else 0.0
                if gap < -1e-5:
                    raise ValueError('A saved heuristic contradicts its proven optimal reference.')
                group['gap'].append(max(0.0, gap))
                group['paired_method'].append(row['e2e_seconds'])
                group['paired_milp'].append(ref)
            for resource in json.loads(row['resources_json']):
                device = resource[0]
                for index, pool in enumerate(POOLS):
                    limit = limits[device][index]
                    if not limit:
                        continue
                    value = 100 * resource[index + 1] / limit
                    group['resources'][pool].append(value)
                    for f in (failure, 'all'):
                        total = device_totals[f, method, device, pool]
                        total[0] += 1; total[1] += value; total[2] = max(total[2], value)
        exported = []
        for (failure, method), group in groups.items():
            stat = group_statistics(group)
            exported.append(dict(failure_count=failure, method=method,
                                 scenario_count=scenario_counts[failure], **stat))
            failure_summaries.append(flat_row(size, configuration, failure, method, stat, scenario_counts[failure]))
        for method in METHODS:
            # Merge one method at a time to bound memory on full D15 studies.
            combined = merge_buckets([groups[f, method] for f in sorted(scenario_counts)])
            stat = group_statistics(combined)
            summaries.append(flat_row(size, configuration, 'all', method, stat, len(scenarios)))
            del combined
        per_device = []
        for (failure, method, device, pool), (count, total, maximum) in device_totals.items():
            row = dict(network_size=size, configuration=name, level=configuration['level'],
                link_seed=configuration['link_seed'], failure_count=failure, method=method,
                device_id=device, pool=pool, feasible_surviving_device_observations=count,
                mean_utilization_percent=total / count, maximum_utilization_percent=maximum)
            device_summaries.append(row)
            if failure == 'all':
                per_device.append(row)
        expected = len(scenarios) * (len(plan['random_seeds']) + 4)
        observed = sum(g['trials'] for g in groups.values())
        datasets.append(dict(network_size=size, configuration=name, level=configuration['level'],
            link_seed=configuration['link_seed'], fingerprint=prepared['effective_fingerprint'],
            selection=dict(mode='all' if plan['max_per_failure_count'] is None else 'sampled',
                expected_trials=expected, saved_trials=observed, complete=observed == expected),
            groups=exported, device_utilization=per_device))
        print(f'  Summarized D={size} {name}: {observed:,}/{expected:,} trials.', flush=True)
        del groups
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'summary_by_failure_count.csv', failure_summaries)
    write_csv(output / 'device_utilization_summary.csv', device_summaries)
    aggregate = across_seeds(summaries + failure_summaries)
    write_csv(output / 'summary_across_link_seeds.csv', aggregate)
    if unresolved:
        write_csv(output / 'unresolved_trials.csv', unresolved)
    else:
        (output / 'unresolved_trials.csv').write_text(
            'network_size,configuration,scenario_id,method,seed,status,feasible,e2e_seconds,'
            'objective_bound_seconds,solver_relative_gap,algorithm_ms\n', encoding='utf-8')
    write_json(output / 'link_rate_plot_data.json', dict(
        status=metadata['status'], study_kind=metadata['study_kind'], definitions=metadata['definitions'],
        datasets=datasets, across_link_seeds=aggregate,
        notes=dict(paired_gap='100*(heuristic_latency-MILP_latency)/MILP_latency, averaged over matched recovered trials. '
                            'This is heuristic excess over MILP, not MILP percentage reduction relative to a heuristic.',
            averaging='Across-network summaries weight each requested link seed equally; baseline is counted once. '
                      'Failure combinations within each network have equal weight; Random includes all seeds.',
            latency='Boxplots and own-feasible means condition on each method own recovery success.',
            resources='Pooled surviving-device usage in recovered trials; absent PSRAM has no observations. '
                      'Per-device means are also provided, separately for each memory pool.',
            timing='Median and IQR across completed attempts, including unsuccessful recovery. '
                   'Unresolved attempts are listed separately.',
            coverage='Incomplete studies are labelled PARTIAL; sampled workflow tests are not full experiments.')))


def save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    # Freeze the solved layout before tight-bbox export so the PDF/PNG backends
    # cannot move a long y-axis label beyond an already calculated crop box.
    fig.canvas.draw()
    fig.set_constrained_layout(False)
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(directory / f'{name}.png', dpi=170, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def figure_legend(fig, plt):
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=c, marker=m, label=l, markersize=4, linewidth=1.2)
                        for c, m, l in zip(COLORS, MARKERS, LABELS)],
               loc='lower center', bbox_to_anchor=(.5, -.095), ncol=3, fontsize=7, frameon=False)


def coverage_label(dataset):
    mode = 'ALL combinations' if dataset['selection']['mode'] == 'all' else 'SAMPLED test'
    if not dataset['selection']['complete']:
        mode += ' - PARTIAL'
    return mode


def title_for(dataset):
    if not dataset['level']:
        return 'Original link rates (baseline)'
    return f'Rate variation ≤ {100*dataset["level"]:g}%; link seed {dataset["link_seed"]}'


def find_group(dataset, method, failure):
    return next(g for g in dataset['groups'] if g['method'] == method and g['failure_count'] == failure)


def box(ax, stats):
    adjusted = [{k: (math.nan if value[k] is None else value[k]) for k in
                 ('mean', 'med', 'q1', 'q3', 'whislo', 'whishi')} | {'fliers': value['fliers']} for value in stats]
    artists = ax.bxp(adjusted, positions=range(len(METHODS)), widths=0.55, showmeans=True,
        patch_artist=True, meanprops=dict(marker='D', markersize=3, markerfacecolor='white', markeredgecolor='black'),
        flierprops=dict(marker='o', markersize=1.5, alpha=.3), medianprops=dict(color='black', linewidth=.9))
    for patch, color in zip(artists['boxes'], COLORS):
        patch.set_facecolor(color); patch.set_alpha(.65)
    ax.set_xticks(range(len(METHODS)), SHORT, fontsize=6.5)
    ax.grid(axis='y', alpha=.22)


def plot_overview(datasets, directory, plt):
    fig, axes = plt.subplots(4, len(datasets), figsize=(3.5*len(datasets), 9.4),
                             layout='constrained', squeeze=False)
    fig.suptitle(title_for(datasets[0]) + '\nSynthetic link-rate sensitivity', fontsize=12)
    for column, dataset in enumerate(datasets):
        failures = sorted({g['failure_count'] for g in dataset['groups']})
        axes[0, column].set_title(f'D={dataset["network_size"]}; {coverage_label(dataset)}', fontsize=8)
        for method, color, marker in zip(METHODS, COLORS, MARKERS):
            groups = [find_group(dataset, method, f) for f in failures]
            axes[0, column].errorbar(failures,
                [g['feasibility_percent'] if g['feasibility_percent'] is not None else math.nan for g in groups],
                yerr=[g['random_seed_rate_sd'] or 0 for g in groups] if method == 'Random' else None,
                color=color, marker=marker, linewidth=1.1, markersize=3, capsize=2)
            axes[1, column].plot(failures, [g['latency']['mean'] for g in groups],
                color=color, marker=marker, linewidth=1.1, markersize=3)
            times = [g['computation_time_ms'] for g in groups]
            axes[2, column].plot(failures, [s['med'] for s in times],
                color=color, marker=marker, linewidth=1.1, markersize=3)
            axes[2, column].fill_between(failures,
                [s['q1'] if s['q1'] is not None else math.nan for s in times],
                [s['q3'] if s['q3'] is not None else math.nan for s in times], color=color, alpha=.1)
            axes[3, column].plot(failures, [g['resources']['sram']['mean'] for g in groups],
                color=color, marker=marker, linewidth=1.1, markersize=3)
        labels = ('Recovered trials (%)', 'Mean E2E latency (s)\nown feasible',
                  'Algorithm time (ms)\nmedian and IQR', 'Mean SRAM usage (%)\nown feasible devices')
        for row, label in enumerate(labels):
            ax = axes[row, column]; ax.set_xticks(failures); ax.grid(alpha=.22)
            if column == 0:
                ax.set_ylabel(label, fontsize=8)
            if row in (0, 3):
                ax.set_ylim(0, 105)
            elif row == 2:
                ax.set_yscale('log')
            else:
                ax.set_ylim(bottom=0)
        axes[-1, column].set_xlabel('Failed devices, F')
    figure_legend(fig, plt)
    fig.supxlabel('Random includes every seed. Conditional latency/memory populations can differ; see paired-gap figure.\n'
                  'PSRAM, model storage and complete OTA usage are shown separately in resource figures.', fontsize=7)
    save(fig, directory, 'four_metrics_vs_failures')


def plot_paired_failures(datasets, directory, plt):
    fig, axes = plt.subplots(len(datasets), 1, figsize=(7.5, 2.2*len(datasets)+.6),
                             layout='constrained', squeeze=False)
    fig.suptitle(title_for(datasets[0]), fontsize=11)
    for dataset, ax in zip(datasets, axes.flat):
        failures = sorted({g['failure_count'] for g in dataset['groups']})
        for method, color, marker in zip(METHODS[1:], COLORS[1:], MARKERS[1:]):
            ax.plot(failures, [find_group(dataset, method, f)['paired_gap_percent']['mean'] for f in failures],
                    color=color, marker=marker, linewidth=1.2, markersize=4)
        ax.set_title(f'D={dataset["network_size"]}; {coverage_label(dataset)}', fontsize=8)
        ax.set_xticks(failures); ax.set_ylabel('Mean latency excess\nover MILP (%)', fontsize=8)
        ax.set_ylim(bottom=0); ax.grid(alpha=.25)
    axes[-1, 0].set_xlabel('Failed devices, F')
    figure_legend(fig, plt)
    fig.supxlabel('Paired recovered trials with a proven-optimal MILP reference; Random includes every seed.\n'
                  '100 × (heuristic − MILP) / MILP. Read alongside recovery feasibility.', fontsize=7)
    save(fig, directory, 'paired_gap_vs_failures')


def plot_boxes(dataset, directory, plt):
    failures = sorted({g['failure_count'] for g in dataset['groups']})
    size = dataset['network_size']
    fig, axes = plt.subplots(math.ceil(len(failures)/2), 2, figsize=(9, 2.5*math.ceil(len(failures)/2)+.7),
                             layout='constrained', squeeze=False)
    fig.suptitle(f'E2E latency - D={size}; {title_for(dataset)}\n{coverage_label(dataset)}', fontsize=11)
    for f, ax in zip(failures, axes.flat):
        stats = [find_group(dataset, m, f)['latency'] for m in METHODS]
        box(ax, stats); ax.set_title(f'F={f}', fontsize=9); ax.set_ylabel('Latency (s)')
        maxima = [max([stat['whishi']] + stat['fliers']) for stat in stats if stat['n']]
        ax.set_ylim(0, 1.18 * max(maxima) if maxima and max(maxima) > 0 else 1)
        if not maxima:
            ax.text(.5, .5, 'No recovered trials', ha='center', transform=ax.transAxes, fontsize=8)
        for i, stat in enumerate(stats):
            ax.text(i, .98, f'n={stat["n"]}', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=6)
    for ax in list(axes.flat)[len(failures):]:
        ax.axis('off')
    fig.supxlabel('Own-feasible trial distributions. Line: median; diamond: mean; whiskers: 1.5 × IQR.\n'
                  'Random contains all seeds; n counts trials. No successful recovery means an empty box.', fontsize=7)
    save(fig, directory, f'latency_boxplots_D{size}')
    fig, axes = plt.subplots(len(failures), len(POOLS), figsize=(12, 2.1*len(failures)+.7),
                             layout='constrained', squeeze=False)
    fig.suptitle(f'Memory utilization - D={size}; {title_for(dataset)}\n{coverage_label(dataset)}', fontsize=11)
    for row, f in enumerate(failures):
        for col, (pool, label) in enumerate(zip(POOLS, POOL_LABELS)):
            ax = axes[row, col]
            stats = [find_group(dataset, m, f)['resources'][pool] for m in METHODS]
            box(ax, stats); ax.set_ylim(0, 105); ax.axhline(100, color='#B43C32', ls='--', lw=.7)
            if not any(stat['n'] for stat in stats):
                ax.text(.5, .5, 'No observations', ha='center', transform=ax.transAxes, fontsize=7)
            if row == 0:
                ax.set_title(label, fontsize=8)
            if col == 0:
                ax.set_ylabel(f'F={f}\nUsage (%)', fontsize=8)
    fig.supxlabel('Pooled surviving-device observations in recovered trials; memory pools use separate capacities.\n'
                  'Line: median; diamond: mean; whiskers: 1.5 × IQR. Devices with zero PSRAM are excluded from PSRAM statistics.', fontsize=7)
    save(fig, directory, f'memory_boxplots_D{size}')


def plot_devices(dataset, directory, plt):
    size = dataset['network_size']
    devices = range(1, size+1)
    fig, axes = plt.subplots(4, 1, figsize=(max(7.5, .53*size), 8.8), layout='constrained')
    fig.suptitle(f'Per-device memory - D={size}; {title_for(dataset)}\n{coverage_label(dataset)}', fontsize=11)
    values = {(r['method'], r['device_id'], r['pool']): r['mean_utilization_percent']
              for r in dataset['device_utilization']}
    for ax, pool, label in zip(axes, POOLS, POOL_LABELS):
        matrix = np.array([[values.get((method, device, pool), math.nan) for device in devices] for method in METHODS])
        artist = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=100, aspect='auto', cmap='YlGnBu')
        ax.set_yticks(range(len(METHODS)), LABELS, fontsize=7)
        ax.set_xticks(range(size), list(devices), fontsize=7); ax.set_title(label, fontsize=8)
        fig.colorbar(artist, ax=ax, label='Mean usage (%)', fraction=.025, pad=.015)
    axes[-1].set_xlabel('Device ID')
    fig.supxlabel('Mean over recovered trials where the device survives, across all selected F.\n'
                  'Blank cells have no observations or no capacity for that pool. Full values and maxima are in device_utilization_summary.csv.', fontsize=7)
    save(fig, directory, f'per_device_memory_D{size}')


def plot_sensitivity(payload, directory, plt):
    rows = [r for r in payload['across_link_seeds'] if r['failure_count'] == 'all']
    sizes = sorted({r['network_size'] for r in rows})
    fig, axes = plt.subplots(2, len(sizes), figsize=(3.6*len(sizes), 6.1), layout='constrained', squeeze=False)
    label = ' - PARTIAL' if any(not d['selection']['complete'] for d in payload['datasets']) else ''
    if 'sampled' in payload['study_kind']:
        label += ' - SAMPLED test'
    fig.suptitle(f'Synthetic link-rate sensitivity{label}', fontsize=12)
    for column, size in enumerate(sizes):
        for method, color, marker in zip(METHODS, COLORS, MARKERS):
            sample = sorted((r for r in rows if r['network_size'] == size and r['method'] == method),
                            key=lambda r: r['level'])
            xs = [100*r['level'] for r in sample]
            for row, field in enumerate(('mean_paired_gap_percent', 'feasibility_percent')):
                ax = axes[row, column]
                ax.plot(xs, [r[f'{field}_layout_mean'] for r in sample],
                        color=color, marker=marker, linewidth=1.3, markersize=4)
                ax.fill_between(xs,
                    [r[f'{field}_layout_min'] if r[f'{field}_layout_min'] is not None else math.nan for r in sample],
                    [r[f'{field}_layout_max'] if r[f'{field}_layout_max'] is not None else math.nan for r in sample],
                    color=color, alpha=.1)
                ax.set_xticks(xs); ax.grid(alpha=.25)
        # Set limits after all methods are drawn; doing this on the zero-gap
        # MILP line would disable autoscaling before the heuristic curves exist.
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].set_ylim(0, 105)
        axes[1, column].set_xlabel('Maximum rate reduction (%)')
        axes[0, column].set_title(f'D={size}', fontsize=10)
        if column == 0:
            axes[0, column].set_ylabel('Mean paired latency excess\nover MILP (%)')
            axes[1, column].set_ylabel('Recovered trials (%)')
    figure_legend(fig, plt)
    fig.supxlabel('Line: equal-weight mean across link seeds; shading: min–max across link seeds, not a confidence interval.\n'
                  'Original baseline counted once. Random includes all its seeds. Missing paired populations are left blank.', fontsize=7)
    save(fig, directory, 'link_rate_sensitivity')


def draw_all(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    payload = json.loads((output / 'link_rate_plot_data.json').read_text())
    directory = output / 'figures'
    plot_sensitivity(payload, directory, plt)
    by_configuration = defaultdict(list)
    for dataset in payload['datasets']:
        by_configuration[dataset['configuration']].append(dataset)
    for name, datasets in by_configuration.items():
        target = directory / name
        datasets.sort(key=lambda d: d['network_size'])
        plot_overview(datasets, target, plt)
        plot_paired_failures(datasets, target, plt)
        for dataset in datasets:
            plot_boxes(dataset, target, plt)
            plot_devices(dataset, target, plt)
        print(f'  Saved PDF/PNG figures: {name}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('link_rate_study'))
    args = parser.parse_args()
    draw_all(args.output.resolve())
    print(f'Figures saved to {(args.output / "figures").resolve()}')
