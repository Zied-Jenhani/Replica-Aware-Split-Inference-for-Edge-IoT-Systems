"""Re-run the existing ICC algorithms on reproducible pair-specific link rates.

Copy this file and plot_link_rate_study.py beside solve_milp.py and
run_heuristics.py. Run: python run_link_rate_study.py --output link_rate_study
Stop with Ctrl+C; add --resume to the SAME command to continue.
"""
from __future__ import annotations

# One solver worker at a time. Set these before importing NumPy/SciPy.
import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '1'

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import shlex
import signal
import sqlite3
import sys
import time
import traceback
import zipfile

import solve_milp as core
import run_heuristics as heuristics

VERSION = 'icc-link-rate-study-v1'
METHODS = ('MILP', 'Random', 'ComputationGreedy', 'SRAMWorstFit', 'FlashFFD')
INPUT_FILES = ('case.json', 'devices.csv', 'segments.csv', 'links.csv')
COMPLETE_STATUSES = ('optimal', 'proven_infeasible', 'feasible', 'heuristic_failed')
ROW_COLUMNS = ('network_size', 'configuration', 'scenario_id', 'method', 'seed',
               'failure_count', 'status', 'feasible', 'completed_attempt',
               'e2e_seconds', 'computation_seconds', 'communication_seconds',
               'algorithm_ms', 'build_ms', 'placement_ms', 'protocol_ms',
               'objective_bound_seconds', 'solver_relative_gap',
               'resources_json', 'placement_json', 'protocols_json')


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes(directory):
    return {name: sha256(directory / name) for name in INPUT_FILES}


def unit_random(*parts):
    # Exactly the same hash convention as the completed link pilots.
    digest = hashlib.sha256(':'.join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64


def vary_rates(links, size, level, seed):
    if level == 0:
        return dict(links)
    return {(i, j, protocol): rate * (1 - level * unit_random(
                seed, size, 'rate', min(i, j), max(i, j), protocol))
            for (i, j, protocol), rate in links.items()}


def runtime_metadata():
    return dict(python=platform.python_version(), numpy=core.np.__version__,
                scipy=core.scipy.__version__, platform=platform.platform(),
                machine=platform.machine(), solver='SciPy milp / HiGHS',
                concurrent_solver_workers=1,
                thread_environment={name: os.environ[name] for name in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})


def source_directory(root, size):
    directory = root / f'inputs_D{size}'
    if size == 5 and not (directory / 'case.json').is_file():
        directory = root / 'inputs'
    missing = [name for name in INPUT_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f'Missing {", ".join(missing)} in {directory}. '
                         'D5 may use inputs/; D10 and D15 need inputs_D10/ and inputs_D15/.')
    return directory.resolve()


def configurations(args):
    return [dict(name='baseline', level=0.0, link_seed=None)] + [
        dict(name=f'rate_{round(100 * level)}_seed_{seed}', level=level, link_seed=seed)
        for level in args.levels for seed in args.link_seeds]


def prepare_plan(args, output):
    """Validate all source sizes before creating any output directory."""
    networks, loaded = {}, {}
    for size in args.sizes:
        directory = source_directory(args.inputs_root, size)
        if output == directory or output in directory.parents or directory in output.parents:
            raise ValueError('The output must be separate from every source input directory.')
        nodes, segments, links, case = core.load_inputs(directory, failed=[], protocol='joint')
        if len(nodes) != size or len(segments) != size:
            raise ValueError(f'{directory} must contain D=K={size}.')
        initial = core.check_placement(nodes, segments, links, case,
                                     {k: s['primary_device_id'] for k, s in segments.items()})
        if not initial['feasible']:
            raise ValueError(f'Original D={size} deployment is infeasible: {initial["errors"]}')
        scenarios = core.failure_scenarios(nodes, segments, case,
                                          args.max_per_failure_count, args.scenario_seed)
        networks[str(size)] = dict(source_directory=str(directory), source_sha256=input_hashes(directory),
            source_effective_fingerprint=core.experiment_metadata(nodes, segments, links, case)['experiment_sha256'],
            input_file_protocol_policy=json.loads((directory / 'case.json').read_text())['protocol_policy'],
            effective_protocol_policy='joint',
            scenarios=[dict(id=sid, failed=failed) for sid, failed in scenarios])
        loaded[size] = nodes, segments, links, case
    scripts = [Path(__file__), Path(core.__file__), Path(heuristics.__file__)]
    return dict(version=VERSION, sizes=args.sizes, configurations=configurations(args),
        random_seeds=list(range(args.random_runs)), scenario_seed=args.scenario_seed,
        max_per_failure_count=args.max_per_failure_count,
        solver_time_limit_seconds=args.time_limit, runtime=runtime_metadata(),
        algorithm_sha256={p.name: sha256(p) for p in scripts}, networks=networks), loaded


def generate_inputs(output, plan, loaded):
    prepared, initial_rows, rate_rows = [], [], []
    for size in plan['sizes']:
        nodes, segments, links, original_case = loaded[size]
        source = Path(plan['networks'][str(size)]['source_directory'])
        for configuration in plan['configurations']:
            name, level, seed = (configuration[k] for k in ('name', 'level', 'link_seed'))
            directory = output / 'generated_inputs' / f'D{size}' / name
            directory.mkdir(parents=True)
            modified = vary_rates(links, size, level, seed)
            case = copy.deepcopy(original_case)
            case['solver']['time_limit'] = plan['solver_time_limit_seconds']
            case['solver']['mip_rel_gap'] = 0.0
            case['failed_device_ids'] = []
            case['protocol_policy'] = 'joint'
            case['link_rate_study'] = dict(configuration=name, level=level, link_seed=seed,
                input_status='SYNTHETIC_LINK_RATE_SENSITIVITY',
                formula='B_new(i,j,p)=B_original(i,j,p)*(1-level*u(seed,D,unordered_pair,p))',
                explanation='Only link rates vary; benchmark values are anchors, not new radio measurements.')
            case.setdefault('notes', []).append(
                'This generated sensitivity configuration supersedes any inherited uniform-rate assumption; '
                'baseline retains the source link rates. Original input files are unchanged.')
            for filename in ('devices.csv', 'segments.csv'):
                (directory / filename).write_bytes((source / filename).read_bytes())
            write_csv(directory / 'links.csv', (
                dict(source_id=i, destination_id=j, protocol=p, B_bytes_per_s=rate)
                for (i, j, p), rate in sorted(modified.items())))
            write_json(directory / 'case.json', case)
            actual_nodes, actual_segments, actual_links, actual_case = core.load_inputs(
                directory, failed=[], protocol='joint')
            if actual_nodes != nodes or actual_segments != segments or actual_links != modified:
                raise RuntimeError('Generated inputs changed device/segment values or link serialization.')
            if set(actual_links) != set(links):
                raise RuntimeError('Rate sensitivity must preserve every original link/protocol.')
            checked = core.check_placement(actual_nodes, actual_segments, actual_links, actual_case,
                                          {k: s['primary_device_id'] for k, s in segments.items()})
            if not checked['feasible']:
                raise RuntimeError(f'{directory}: generated initial placement is infeasible.')
            relative = directory.relative_to(output).as_posix()
            prepared.append(dict(network_size=size, configuration=name, directory=relative,
                input_sha256=input_hashes(directory),
                effective_fingerprint=core.experiment_metadata(actual_nodes, actual_segments,
                    actual_links, actual_case)['experiment_sha256']))
            initial_rows.append(dict(network_size=size, configuration=name, level=level, link_seed=seed,
                initial_feasible=True, initial_e2e_seconds=checked['objective_seconds'],
                initial_computation_seconds=checked['computation_seconds'],
                initial_communication_seconds=checked['communication_seconds'],
                directed_pairs=len({(i, j) for i, j, p in actual_links}),
                directed_protocol_entries=len(actual_links)))
            for protocol in sorted({p for i, j, p in links}):
                ratios = [modified[key] / rate for key, rate in links.items() if key[2] == protocol]
                rate_rows.append(dict(network_size=size, configuration=name, protocol=protocol,
                    link_seed=seed, level=level, directed_entries=len(ratios),
                    minimum_multiplier=min(ratios), mean_multiplier=sum(ratios) / len(ratios),
                    maximum_multiplier=max(ratios)))
    write_csv(output / 'initial_deployments.csv', initial_rows)
    write_csv(output / 'link_rate_summary.csv', rate_rows)
    return prepared


def open_database(output):
    database = sqlite3.connect(output / 'study.sqlite')
    database.row_factory = sqlite3.Row
    database.execute('PRAGMA journal_mode=WAL')
    database.execute('PRAGMA synchronous=NORMAL')
    database.execute('''CREATE TABLE IF NOT EXISTS trials (
        network_size INTEGER NOT NULL, configuration TEXT NOT NULL,
        scenario_id TEXT NOT NULL, method TEXT NOT NULL, seed INTEGER NOT NULL,
        failure_count INTEGER NOT NULL, status TEXT NOT NULL,
        feasible INTEGER NOT NULL, completed_attempt INTEGER NOT NULL,
        e2e_seconds REAL, computation_seconds REAL, communication_seconds REAL,
        algorithm_ms REAL NOT NULL, build_ms REAL, placement_ms REAL, protocol_ms REAL,
        objective_bound_seconds REAL, solver_relative_gap REAL,
        resources_json TEXT NOT NULL, placement_json TEXT NOT NULL, protocols_json TEXT NOT NULL,
        PRIMARY KEY (network_size, configuration, scenario_id, method, seed)
    ) WITHOUT ROWID''')
    database.commit()
    return database


def pack_report(report):
    # Keep raw resource usage and placements locally without thousands of folders.
    resources = [[r['device_id']] + [r[f'{p}_used_bytes'] for p in ('model', 'flash', 'sram', 'psram')]
                 for r in report.get('resources', [])] if report['feasible'] else []
    packed = dict(status=report['status'], feasible=int(report['feasible']),
        completed_attempt=int(report['status'] in COMPLETE_STATUSES),
        e2e_seconds=report.get('objective_seconds'),
        computation_seconds=report.get('computation_seconds'),
        communication_seconds=report.get('communication_seconds'),
        objective_bound_seconds=report.get('objective_bound_seconds'),
        solver_relative_gap=report.get('solver_reported_relative_gap'),
        resources_json=json.dumps(resources, separators=(',', ':')),
        placement_json=json.dumps(sorted(report.get('placement', {}).items()), separators=(',', ':')),
        protocols_json=json.dumps(dict(Counter(t['protocol'] for t in report.get('transfers', []))),
                                  sort_keys=True, separators=(',', ':')))
    for column, field in (('algorithm_ms', 'algorithm_seconds'), ('build_ms', 'build_seconds'),
                          ('placement_ms', 'placement_seconds'), ('protocol_ms', 'protocol_optimization_seconds')):
        packed[column] = 1000 * report[field] if field in report else None
    return packed


def worker_main(connection, directory):
    # Ctrl+C is handled by the parent, which terminates this process immediately.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        nodes, segments, links, template = core.load_inputs(directory, failed=[], protocol='joint')
        connection.send(('ready', None))
        while True:
            job = connection.recv()
            if job is None:
                break
            sid, failed, method, seed = job
            case = {**template, 'failed_device_ids': failed}
            if method == 'MILP':
                report, _ = core.solve(nodes, segments, links, case)
            else:
                report, _ = heuristics.run_case(nodes, segments, links, case, method,
                                                None if seed == -1 else seed, sid)
            connection.send(('result', pack_report(report)))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class SolverWorker:
    def __init__(self, directory):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker_main, args=(child, str(directory)), daemon=True)
        self.process.start()
        child.close()

    def receive(self, label):
        began, last_notice = time.monotonic(), time.monotonic()
        while not self.connection.poll(0.2):
            if not self.process.is_alive():
                raise RuntimeError(f'Solver worker exited during {label}; committed trials are safe.')
            if time.monotonic() - last_notice >= 25:
                print(f'  Still solving {label} ({time.monotonic()-began:.0f}s). Ctrl+C stops and saves progress.', flush=True)
                last_notice = time.monotonic()
        try:
            tag, payload = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError(f'Solver worker disconnected during {label}.') from exc
        if tag == 'error':
            raise RuntimeError(payload)
        return tag, payload

    def solve(self, job):
        self.connection.send(job)
        tag, result = self.receive(f'{job[2]} / {job[0]}')
        if tag != 'result':
            raise RuntimeError('Unexpected solver-worker response.')
        return result

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)
        self.connection.close()


def validate_reference(database, size, config, sid, method, row):
    if method == 'MILP':
        return
    reference = database.execute('SELECT status, e2e_seconds FROM trials WHERE '
        'network_size=? AND configuration=? AND scenario_id=? AND method="MILP" AND seed=-1',
        (size, config, sid)).fetchone()
    if reference is None:
        raise RuntimeError('A heuristic trial has no matching MILP reference.')
    if row['feasible'] and reference['status'] == 'proven_infeasible':
        raise RuntimeError('A verified heuristic contradicts the MILP infeasibility proof.')
    if row['feasible'] and reference['status'] == 'optimal':
        if row['e2e_seconds'] < reference['e2e_seconds'] - max(1e-8, 1e-7 * abs(reference['e2e_seconds'])):
            raise RuntimeError('A heuristic beats the proven MILP optimum; inspect the implementation.')


def run_trials(database, output, metadata):
    plan = metadata['plan']
    total = metadata['expected_trials']
    done = database.execute('SELECT COUNT(*) FROM trials').fetchone()[0]
    last_notice = time.monotonic()
    placeholders = ','.join('?' for _ in ROW_COLUMNS)
    insert = f'INSERT INTO trials ({",".join(ROW_COLUMNS)}) VALUES ({placeholders})'
    jobs = [('MILP', -1)] + [('Random', seed) for seed in plan['random_seeds']] + [
        (method, -1) for method in METHODS[2:]]
    for prepared in metadata['prepared_inputs']:
        size, config = prepared['network_size'], prepared['configuration']
        scenarios = plan['networks'][str(size)]['scenarios']
        condition_total = len(scenarios) * len(jobs)
        condition_done = database.execute('SELECT COUNT(*) FROM trials WHERE '
            'network_size=? AND configuration=?', (size, config)).fetchone()[0]
        if condition_done == condition_total:
            print(f'D={size} {config}: already complete; skipped.', flush=True)
            continue
        print(f'D={size} {config}: {len(scenarios):,} scenarios, '
              f'{condition_done:,}/{condition_total:,} trials saved.', flush=True)
        worker = SolverWorker(output / prepared['directory'])
        try:
            if worker.receive('worker initialization')[0] != 'ready':
                raise RuntimeError('Worker did not initialize.')
            for scenario in scenarios:
                sid, failed = scenario['id'], scenario['failed']
                stored = {(r['method'], r['seed']) for r in database.execute(
                    'SELECT method,seed FROM trials WHERE network_size=? AND configuration=? AND scenario_id=?',
                    (size, config, sid))}
                for method, seed in jobs:
                    if (method, seed) in stored:
                        continue
                    row = worker.solve((sid, failed, method, seed))
                    validate_reference(database, size, config, sid, method, row)
                    row.update(network_size=size, configuration=config, scenario_id=sid,
                               method=method, seed=seed, failure_count=len(failed))
                    with database:
                        database.execute(insert, [row[k] for k in ROW_COLUMNS])
                    done += 1
                    condition_done += 1
                    if time.monotonic() - last_notice >= 10 or condition_done == condition_total:
                        print(f'  {done:,}/{total:,} total trials ({100*done/total:.2f}%) | '
                              f'D={size} {config}: {condition_done:,}/{condition_total:,} | {sid}', flush=True)
                        last_notice = time.monotonic()
        finally:
            worker.close()


def verify_inputs(metadata, output):
    for network in metadata['plan']['networks'].values():
        if input_hashes(Path(network['source_directory'])) != network['source_sha256']:
            raise ValueError('Source inputs changed since preparation. Use a new output folder for new data.')
    for prepared in metadata['prepared_inputs']:
        if input_hashes(output / prepared['directory']) != prepared['input_sha256']:
            raise ValueError(f'Generated inputs changed: {prepared["directory"]}. Use a new output folder.')


def make_bundle(output):
    # The database stays on the benchmark PC; figures + exact inputs + compact
    # statistics are sufficient to review without sending gigabytes of trials.
    name = output / 'link_rate_results_bundle.zip'
    temporary = name.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        files = sorted(p for p in output.iterdir() if p.is_file() and p.suffix in ('.csv', '.json'))
        files += sorted((output / 'generated_inputs').rglob('*.csv'))
        files += sorted((output / 'generated_inputs').rglob('*.json'))
        files += sorted((output / 'figures').rglob('*.pdf'))
        files += sorted((output / 'figures').glob('*.png'))
        for path in files:
            archive.write(path, path.relative_to(output))
    temporary.replace(name)
    print(f'\nSend back: {name} ({name.stat().st_size / 1024**2:.1f} MiB)', flush=True)


def output_lock(output):
    # Advisory locks release automatically on crashes; stale files cannot block resume.
    import fcntl
    lock = (output / 'runner.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise ValueError('Another runner is using this output folder. Stop it before resuming.') from exc
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inputs-root', type=Path, default=Path('.'))
    parser.add_argument('--sizes', type=int, nargs='+', choices=(5, 10, 15), default=[5, 10, 15])
    parser.add_argument('--levels', type=float, nargs='+', choices=(0.25, 0.5, 0.75), default=[0.25, 0.5, 0.75])
    parser.add_argument('--link-seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--random-runs', type=int, default=30)
    parser.add_argument('--time-limit', type=float, default=60.0)
    parser.add_argument('--max-per-failure-count', type=int, default=None,
                        help='Optional SAMPLE for testing only; omit for the complete paper study.')
    parser.add_argument('--scenario-seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, default=Path('link_rate_study'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Create and validate inputs; run no algorithms.')
    parser.add_argument('--summarize-only', action='store_true', help='Rebuild summaries and figures from saved trials.')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.random_runs < 1 or not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error('Random runs and the finite solver time limit must be positive.')
    if args.max_per_failure_count is not None and args.max_per_failure_count < 1:
        parser.error('The sampled scenario cap must be positive.')
    for name in ('sizes', 'levels', 'link_seeds'):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f'--{name.replace("_", "-")} must not contain duplicates.')
    args.sizes.sort(); args.levels.sort(); args.link_seeds.sort()
    output = args.output.resolve()
    manifest = output / 'study_metadata.json'
    if args.summarize_only:
        if not manifest.is_file():
            parser.error('No study_metadata.json exists in the selected output directory.')
        plan, loaded = None, None
    else:
        if output.exists() and not args.resume:
            parser.error(f'{output} already exists. Add --resume to continue it, or choose a new --output name.')
        if args.resume and not manifest.is_file():
            parser.error('--resume needs an existing study_metadata.json; use a fresh output name otherwise.')
        import plot_link_rate_study  # Check the companion/NumPy before expensive runs.
        if not args.no_plots and not args.prepare_only:
            import matplotlib  # Fail before running if plotting dependencies are unavailable.
        plan, loaded = prepare_plan(args, output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output_lock(output)
    database = None
    try:
        if manifest.is_file():
            metadata = json.loads(manifest.read_text())
            if metadata['plan']['version'] != VERSION:
                raise ValueError('This output was created by a different runner format.')
            if not args.summarize_only and metadata['plan'] != plan:
                raise ValueError('Resume settings, inputs, algorithm files or Python/runtime differ. '
                                 'Use the original command/environment, or choose a new output folder.')
            verify_inputs(metadata, output)
        else:
            prepared = generate_inputs(output, plan, loaded)
            expected = sum(len(n['scenarios']) for n in plan['networks'].values()) * len(
                plan['configurations']) * (len(plan['random_seeds']) + 4)
            metadata = dict(status='prepared', plan=plan, prepared_inputs=prepared,
                expected_trials=expected, completed_trials=0, active_elapsed_seconds=0.0,
                study_kind='synthetic_full_failure_enumeration' if args.max_per_failure_count is None
                    else 'synthetic_sampled_workflow_test',
                definitions=dict(
                    rates='Positive pair/protocol-specific rates: B_new=B_original*(1-level*u), '
                          'with the same deterministic reciprocal multiplier as the pilots. No links are removed.',
                    assumptions='Synthetic sensitivity around original rate anchors; these are not measured new channels.',
                    scope='Only links.csv rates change; original devices, segments, surviving primaries, '
                          'memory policies and baseline algorithms are retained. Joint protocol policy is explicit.',
                    levels='0.25, 0.50, 0.75 bound reductions by 25%, 50%, 75%; '
                           'individual links have different reductions between zero and that bound.',
                    feasibility='Rate changes alone preserve mathematical feasibility without deadlines or energy limits. '
                                'Verified incumbents count as recovered; unresolved solver statuses are reported separately.',
                    timing='One solver worker; BLAS/OpenMP thread environment set to 1. HiGHS defaults retained. '
                           'Algorithm time includes construction, '
                           'optimization and verification; excludes process startup, input loading, IPC, checkpointing and plots.',
                    comparisons='Latency gaps compare each recovered heuristic trial with the proven-optimal MILP '
                                'for the same size, configuration and failure scenario. All Random seeds are included.',
                    seeds='All requested link seeds/levels are evaluated; the uniform baseline is run only once.',
                    source='The source cannot fail; F=1..min(5,D-1), independently enumerated, not cumulative failures.',
                    results='Raw trials and resource bytes are checkpointed in study.sqlite. '
                            'The shareable ZIP contains statistics, exact generated inputs and figures.'))
            write_json(manifest, metadata)
        print(f'Output: {output}', flush=True)
        for size in metadata['plan']['sizes']:
            print(f'D={size}: {len(metadata["plan"]["networks"][str(size)]["scenarios"]):,} failure combinations.', flush=True)
        print(f'{len(metadata["plan"]["configurations"])} link configurations; '
              f'{len(metadata["plan"]["random_seeds"])} Random seeds; '
              f'{metadata["expected_trials"]:,} total algorithm trials.', flush=True)
        if args.prepare_only:
            print('Inputs prepared and validated. Add --resume to this command and remove --prepare-only to run.', flush=True)
            return 0
        database = open_database(output)
        began = time.monotonic()
        if not args.summarize_only:
            metadata['status'] = 'running'
            write_json(manifest, metadata)
            try:
                run_trials(database, output, metadata)
            except BaseException as exc:
                metadata['active_elapsed_seconds'] += time.monotonic() - began
                metadata['completed_trials'] = database.execute('SELECT COUNT(*) FROM trials').fetchone()[0]
                metadata['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
                metadata['last_error'] = str(exc)
                write_json(manifest, metadata)
                if isinstance(exc, KeyboardInterrupt):
                    resume_args = [a for a in sys.argv if a not in ('--resume', '--prepare-only', '--summarize-only')]
                    print(f'\nStopped. {metadata["completed_trials"]:,} committed trials saved.\nResume:\n'
                          + shlex.join([sys.executable] + resume_args + ['--resume']), flush=True)
                    return 130
                raise
            metadata['active_elapsed_seconds'] += time.monotonic() - began
            metadata.pop('last_error', None)
            verify_inputs(metadata, output)
        metadata['completed_trials'] = database.execute('SELECT COUNT(*) FROM trials').fetchone()[0]
        metadata['unresolved_trials'] = database.execute('SELECT COUNT(*) FROM trials WHERE completed_attempt=0').fetchone()[0]
        all_done = metadata['completed_trials'] == metadata['expected_trials']
        metadata['status'] = ('finished_with_unresolved' if metadata['unresolved_trials'] else 'finished') if all_done else 'partial'
        write_json(manifest, metadata)
        print('Building compact summaries and figures from saved trials...', flush=True)
        import plot_link_rate_study as plotting
        plotting.build_summaries(database, output, metadata)
        if not args.no_plots:
            plotting.draw_all(output)
        make_bundle(output)
        print(f'Status: {metadata["status"]}; unresolved trials: {metadata["unresolved_trials"]:,}.', flush=True)
        return 0
    finally:
        if database is not None:
            database.close()
        lock.close()


def stop_signal(signum, frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop_signal)
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError, sqlite3.Error, ImportError) as exc:
        raise SystemExit(f'Study stopped: {exc}')
