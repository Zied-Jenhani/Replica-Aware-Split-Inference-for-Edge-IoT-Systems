"""Four replica-placement baselines for K=D=5/10/15/20.

Keep beside solve_milp.py and your existing inputs/ directory. Run:
    python run_heuristics.py --random-runs 30

By default, all F=1,...,min(5,D-1) independent failure sets are used.
Use the same --max-per-failure-count and --scenario-seed as the MILP for a
clearly labeled sampled pilot. Never select the best Random seed afterward.
Surviving primaries stay fixed. Every candidate uses the SAME model quota,
OTA Flash, SRAM, PSRAM and directed-link checks as the MILP.

Paper baselines:
  Random: shuffled failed segments; uniformly random eligible surviving host.
  ComputationGreedy: pipeline order; minimum segment computation latency.
  SRAMWorstFit: pipeline order; greatest remaining SRAM fraction AFTER placement.
  FlashFFD: descending model-file size; first eligible ascending device ID.
Deterministic ties use ascending device ID. These methods do not backtrack.
Eligibility checks links to already assigned neighbors; it does not predict
unassigned neighbors. SRAMWorstFit ranks actual SRAM, also under the ESP32
PSRAM arena policy; it never silently substitutes PSRAM in the ranking.

After placement, select the fastest compatible protocol independently at each
boundary and verify the complete placement. For the CURRENT latency/memory
model this exactly minimizes communication latency for fixed hosts: links
have no shared energy, capacity, interference, or deadline constraints.
No second MILP is built or solved. Fixed-host protocol optimality is NOT a
globally optimal placement claim. A greedy failure is not global infeasibility.

Random uses seeds 0,...,random_runs-1 by default, with an independent random
stream for each (seed, scenario). Report every trial, never the best seed.
Algorithm time includes construction, placement checks, protocol selection,
and final verification, but excludes imports, input loading and output writing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import solve_milp as core

ROOT = Path(__file__).resolve().parent
METHODS = ('Random', 'ComputationGreedy', 'SRAMWorstFit', 'FlashFFD')
IMPLEMENTATION_VERSION = 'direct-protocol-selection-v1'
SUPPORTED_MODEL_SCHEMA = 'icc-model-quota-v1'


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def construct_placement(devices, segments, links, case, method, seed, scenario):
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    failed = set(case['failed_device_ids'])
    survivors = sorted(set(devices) - failed)
    placement = {k: s['primary_device_id'] for k, s in segments.items()
                 if s['primary_device_id'] not in failed}
    missing = sorted(set(segments) - set(placement))
    rng = random.Random(f'{seed}:{scenario}')
    if method == 'Random':
        rng.shuffle(missing)
    elif method == 'FlashFFD':
        missing.sort(key=lambda k: (-segments[k]['m_k_bytes'], k))
    trace = []
    for k in missing:
        candidates, rejected = {}, {}
        for d in survivors:
            checked = core.check_placement(devices, segments, links, case,
                                           {**placement, k: d}, partial=True)
            if checked['feasible']:
                candidates[d] = checked
            else:
                rejected[d] = checked['errors']
        step = dict(segment_id=k, eligible_device_ids=list(candidates), rejections=rejected)
        if not candidates:
            step['selected_device_id'] = None
            trace.append(step)
            return None, placement, trace
        if method == 'Random':
            selected = rng.choice(list(candidates))
        elif method == 'ComputationGreedy':
            selected = min(candidates, key=lambda d: (core.compute_seconds(segments[k], devices[d], case), d))
        elif method == 'SRAMWorstFit':
            def remaining_fraction(d):
                row = next(r for r in candidates[d]['resources'] if r['device_id'] == d)
                return (row['sram_limit_bytes'] - row['sram_used_bytes']) / row['sram_limit_bytes']
            selected = min(candidates, key=lambda d: (-remaining_fraction(d), d))
        else:  # FlashFFD
            selected = min(candidates)
        placement[k] = selected
        step['selected_device_id'] = selected
        trace.append(step)
    return placement, placement, trace


def run_case(devices, segments, links, case, method, seed, scenario):
    started = time.perf_counter()
    if core.SCHEMA_VERSION != SUPPORTED_MODEL_SCHEMA:
        raise ValueError('Direct protocol selection requires the current independent-link latency/memory model.')
    placement, partial, trace = construct_placement(devices, segments, links, case, method, seed, scenario)
    placement_seconds = time.perf_counter() - started
    # Empty audit model: no placement/protocol optimization model is built here.
    # Keep the return signature compatible with the existing runners/report writer.
    model = core.LinearModel()
    verification_seconds = 0.0
    if placement is None:
        report = dict(status='heuristic_failed', feasible=False,
                      message='No eligible host at a greedy step; this is not a global infeasibility proof.',
                      partial_placement=partial, protocol_solver_status='not_run',
                      protocol_selection_status='not_run',
                      build_seconds=0.0, variable_counts=dict(x_binary=0, y_continuous=0, z_binary=0),
                      constraint_count=0, variables=[])
        protocol_seconds = 0.0
    else:
        protocol_started = time.perf_counter()
        # With hosts fixed, computation and all memory charges are constants.
        # Each independent transfer minimizes activation_bytes / throughput.
        # The common checker respects both joint and fixed-protocol policies,
        # and represents colocated segments with a zero-cost LOCAL transfer.
        selected = core.check_placement(devices, segments, links, case, placement)
        protocol_seconds = time.perf_counter() - protocol_started
        if not selected['feasible'] or not selected['complete']:
            raise RuntimeError('Complete heuristic placement failed verification: ' + '; '.join(selected['errors']))

        verification_started = time.perf_counter()
        protocols = {transfer['boundary']: transfer['protocol'] for transfer in selected['transfers']}
        checked = core.check_placement(devices, segments, links, case, placement, protocols=protocols)
        if not checked['feasible'] or not checked['complete']:
            raise RuntimeError('Direct protocol selection failed final verification: ' + '; '.join(checked['errors']))
        for field in ('objective_seconds', 'computation_seconds', 'communication_seconds'):
            if not math.isclose(selected[field], checked[field], rel_tol=1e-7, abs_tol=1e-8):
                raise RuntimeError('Direct protocol selection disagrees with final latency verification')
        verification_seconds = time.perf_counter() - verification_started
        report = dict(checked, status='feasible',
                      message='Optimal independent protocol selection for the fixed heuristic hosts; no MILP invoked.',
                      protocol_solver_status='not_run',
                      protocol_selection_status='optimal_for_fixed_hosts',
                      independent_verification_passed=True,
                      build_seconds=0.0, variable_counts=dict(x_binary=0, y_continuous=0, z_binary=0),
                      constraint_count=0, variables=[],
                      solver_reported_relative_gap=None, objective_bound_seconds=None,
                      full_objective_relative_gap=None)
    report.update(
        method=method, seed=seed, scenario_id=scenario,
        failed_device_ids=list(case['failed_device_ids']),
        protocol_policy=case['protocol_policy'], memory_policy=case['memory_policy'],
        placement_trace=trace, placement_seconds=placement_seconds,
        protocol_optimization_seconds=protocol_seconds,
        verification_seconds=verification_seconds,
        protocol_selection_method='direct_fastest_compatible_per_boundary',
        heuristic_implementation_version=IMPLEMENTATION_VERSION,
        optimization_scope='heuristic_placement_then_protocols_for_fixed_hosts',
        global_placement_optimality_claimed=False,
        algorithm_seconds=time.perf_counter() - started,
    )
    return report, model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'inputs')
    parser.add_argument('--output', type=Path, default=ROOT / 'output_heuristics')
    parser.add_argument('--protocol', choices=['joint', 'BLE', 'UDP', 'TCP', 'ESP-NOW'], default='joint')
    parser.add_argument('--memory', choices=['esp32_psram', 'paper_sram'])
    parser.add_argument('--esp32-model-mib', help='ESP32 model quota in MiB (default 1.3)')
    parser.add_argument('--random-runs', type=int, default=30, help='Number of independently seeded Random trials per scenario (default 30)')
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--max-per-failure-count', type=int,
                        help='Pilot: sample up to N failures for each F (default: ALL paper combinations)')
    parser.add_argument('--scenario-seed', type=int, default=2026)
    parser.add_argument('--audit', action='store_true',
                        help='Write full constraints/variables CSV per run; large runs use substantial disk')
    args = parser.parse_args()
    try:
        if args.random_runs < 1:
            raise ValueError('--random-runs must be at least 1')
        devices, segments, links, template = core.load_inputs(
            args.inputs, failed=[], protocol=args.protocol, memory=args.memory,
            esp32_model_mib=args.esp32_model_mib)
        if len(devices) != len(segments) or len(devices) not in (5, 10, 15, 20):
            raise ValueError('Heuristics require D=K in {5,10,15,20}')
        scenarios = core.failure_scenarios(devices, segments, template,
                    args.max_per_failure_count, args.scenario_seed)
        initial = core.check_placement(devices, segments, links, template,
                                      {k: s['primary_device_id'] for k, s in segments.items()})
        if not initial['feasible']:
            raise ValueError('Initial deployment violates the selected budget/policy: ' + '; '.join(initial['errors']))
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = dict(
            **core.experiment_metadata(devices, segments, links, template),
            batch_status='running', methods=list(METHODS),
            random_seeds=list(range(args.seed_start, args.seed_start + args.random_runs)),
            source_device_id=template['source_device_id'],
            network_size=len(devices),
            scenario_selection=dict(mode='all' if args.max_per_failure_count is None else 'sampled_pilot',
                                    max_per_failure_count=args.max_per_failure_count,
                                    scenario_seed=args.scenario_seed),
            protocol_policy=template['protocol_policy'], memory_policy=template['memory_policy'],
            heuristic_implementation_version=IMPLEMENTATION_VERSION,
            protocol_selection_method='direct_fastest_compatible_per_boundary',
            input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.inputs.iterdir()) if p.suffix in ('.csv', '.json')},
            script_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), Path(core.__file__))},
            scenario_ids=[sid for sid, _ in scenarios], reports=[],
            notes=[
                'All trials start from the original primaries. A greedy failure is not a global infeasibility proof.',
                'All methods share the same quota, memory policy and link availability checks.',
                'SRAMWorstFit ranks remaining actual SRAM, including when ESP32 arenas use PSRAM.',
                'Protocol optimization fixes all hosts; optimal protocol choice does not imply optimal placement.',
                'Each Random seed covers the same complete selected scenario list; no best-of-seeds selection.',
                'Random streams are seeded with the string seed:scenario_id using Python random.Random.',
                'Timing includes placement, direct optimal fixed-host protocol selection and final verification; input/output and imports are excluded.',
                'Protocol choices are independent in this model: no energy budget, shared-link capacity or interference coupling.',
            ],
        )
        write_json(args.output / 'run_metadata.json', metadata)
        summary = []
        for method in METHODS:
            seeds = metadata['random_seeds'] if method == 'Random' else [None]
            successful = 0
            for seed in seeds:
                for scenario, failed in scenarios:
                    case = copy.deepcopy(template)
                    case['failed_device_ids'] = failed
                    report, model = run_case(devices, segments, links, case, method, seed, scenario)
                    report.update(core.experiment_metadata(devices, segments, links, case))
                    report['initial_deployment_feasible'] = True
                    report['initial_deployment_errors'] = []
                    relative = Path(method) / (f'seed_{seed}' if seed is not None else 'deterministic') / scenario
                    core.save_report(args.output / relative, report, model, devices, segments, args.inputs, case,
                                     audit=args.audit)
                    metadata['reports'].append(dict(method=method, seed=seed, scenario_id=scenario,
                                                    path=(relative / 'solution.json').as_posix()))
                    summary.append(dict(method=method, seed=seed, scenario_id=scenario,
                                        failure_count=len(failed), status=report['status'], feasible=report['feasible'],
                                        protocol_solver_status=report['protocol_solver_status'],
                                        e2e_seconds=report.get('objective_seconds'),
                                        algorithm_seconds=report['algorithm_seconds']))
                    successful += int(report['feasible'])
            print(f'{method}: {successful}/{len(seeds) * len(scenarios)} feasible trials')
        metadata['batch_status'] = 'finished'
        metadata['run_count'] = len(summary)
        core.write_csv(args.output / 'summary.csv', summary)
        write_json(args.output / 'run_metadata.json', metadata)
        print('Saved:', args.output.resolve())
        print('Next: python calculate_metrics.py')
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Heuristic run failed: {exc}\n')


if __name__ == '__main__':
    main()
