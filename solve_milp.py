"""Replica-placement and protocol MILP for D=K=5, 10, 15, or 20.

Same paper Eqs. (10)-(20), with joint protocol selection, ESP32 PSRAM policy,
and the previously agreed separate 1.3 MiB ESP32 model-storage budget.
Only NumPy and SciPy/HiGHS are required.
"""
from __future__ import annotations
import argparse
import csv
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 'icc-model-quota-v1'
DEFAULT_ESP32_MODEL_MIB = '1.3'


def write_csv(path, rows, fields=None):
    if fields is None and not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(directory, failed=None, protocol=None, memory=None, esp32_model_mib=None):
    """Read the original four input files, without requiring CSV edits.

    ESP32 model storage defaults to floor(1.3 MiB) = 1,363,148 bytes.
    This quota is separate from firmware, physical OTA Flash, and the arena.
    CLI esp32_model_mib overrides case.json's optional esp32_model_budget_bytes.
    For other devices, an optional CSV M_model_budget_bytes is respected;
    otherwise the quota is OTA capacity minus baseline firmware.
    """
    directory = Path(directory)
    def read(name):
        with (directory / name).open(newline='', encoding='utf-8-sig') as stream:
            return list(csv.DictReader(stream))
    case = json.loads((directory / 'case.json').read_text())
    if failed is not None:
        case['failed_device_ids'] = failed
    if protocol is not None:
        case['protocol_policy'] = protocol
    if memory is not None:
        case['memory_policy'] = memory
    if esp32_model_mib is not None or 'esp32_model_budget_bytes' not in case:
        try:
            mib = Decimal(str(esp32_model_mib if esp32_model_mib is not None else DEFAULT_ESP32_MODEL_MIB).replace(',', '.'))
            if not mib.is_finite() or mib < 0:
                raise ValueError('ESP32 model budget must be finite and nonnegative')
            case['esp32_model_budget_bytes'] = int((mib * 1048576).to_integral_value(rounding=ROUND_FLOOR))
        except InvalidOperation as exc:
            raise ValueError('Invalid ESP32 model budget in MiB') from exc
    quota = case['esp32_model_budget_bytes']
    if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
        raise ValueError('esp32_model_budget_bytes must be a nonnegative integer')
    devices = {}
    for row in read('devices.csv'):
        for name in row:
            if name.endswith('_bytes') or name == 'device_id':
                row[name] = int(row[name])
        row['phi_bytes_per_s'] = float(row['phi_bytes_per_s'])
        row.setdefault('M_model_budget_bytes', max(0, row['M_OTA_bytes'] - row['M_base_bytes']))
        if row['device_type'].upper().startswith('ESP32'):
            row['M_model_budget_bytes'] = quota
        if row['device_id'] in devices:
            raise ValueError('Duplicate device ID')
        devices[row['device_id']] = row
    segments = {}
    for row in read('segments.csv'):
        row = {name: int(value) for name, value in row.items()}
        row['u_k_bytes'] = row['u_act_bytes'] + row['u_scr_bytes'] + row['u_state_bytes']
        if row['segment_id'] in segments:
            raise ValueError('Duplicate segment ID')
        segments[row['segment_id']] = row
    links = {}
    for row in read('links.csv'):
        key = int(row['source_id']), int(row['destination_id']), row['protocol']
        if key in links:
            raise ValueError('Duplicate directed protocol link')
        links[key] = float(row['B_bytes_per_s'])
    validate_inputs(devices, segments, links, case)
    return devices, segments, links, case


def experiment_metadata(devices, segments, links, case):
    """Fingerprint effective inputs, including the quota applied in Python.

    Excludes the failure set so all scenarios of one experiment share an ID.
    Prevents metrics from mixing old unlimited-budget runs with new results.
    """
    settings = {k: case[k] for k in ('source_device_id', 'omega_Theta',
                'protocol_policy', 'memory_policy', 'solver', 'esp32_model_budget_bytes')}
    payload = dict(schema_version=SCHEMA_VERSION, devices=devices, segments=segments,
                   links=[[*key, rate] for key, rate in sorted(links.items())], settings=settings)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(schema_version=SCHEMA_VERSION, experiment_sha256=digest,
                effective_model_budgets_bytes={d: row['M_model_budget_bytes'] for d, row in devices.items()})


def failure_scenarios(devices, segments, case, max_per_failure_count=None, scenario_seed=2026):
    """Independent F=1..min(5,D-1); optional reproducible uniform subset per F.

    Use max_per_failure_count=None for ALL combinations as in the paper.
    A sampled subset is only a pilot experiment and must be labeled as such.
    """
    if len(devices) != len(segments) or len(devices) < 2:
        raise ValueError('The network requires D=K >= 2')
    if max_per_failure_count is not None and max_per_failure_count < 1:
        raise ValueError('max_per_failure_count must be positive')
    eligible = sorted(set(devices) - {case['source_device_id']})
    cases = []
    for n in range(1, min(5, len(eligible)) + 1):
        group = list(itertools.combinations(eligible, n))
        if max_per_failure_count is not None and len(group) > max_per_failure_count:
            chosen = sorted(random.Random(f'{scenario_seed}:{len(devices)}:{n}').sample(group, max_per_failure_count))
        else:
            chosen = group
        cases.extend((f'F{n}_' + '-'.join(map(str, failed)), list(failed)) for failed in chosen)
    return cases


def validate_inputs(devices, segments, links, case):
    if not devices or not segments:
        raise ValueError('Device and segment tables cannot be empty')
    if set(segments) != set(range(1, len(segments) + 1)):
        raise ValueError('Segments must have IDs 1,...,K in pipeline order')
    failed = case['failed_device_ids']
    if len(set(failed)) != len(failed) or not set(failed) <= set(devices):
        raise ValueError('Failure list must contain distinct existing device IDs')
    if case['source_device_id'] not in devices or case['source_device_id'] in failed:
        raise ValueError('The data-acquisition device must exist and remain active')
    if case['omega_Theta'] <= 0:
        raise ValueError('omega_Theta must be positive')
    if case['memory_policy'] not in ('esp32_psram', 'paper_sram'):
        raise ValueError('Unknown memory policy')
    if case['protocol_policy'] not in ('BLE', 'UDP', 'TCP', 'ESP-NOW', 'joint'):
        raise ValueError('Unknown protocol policy')
    for d, row in devices.items():
        if row['phi_bytes_per_s'] <= 0 or not math.isfinite(row['phi_bytes_per_s']):
            raise ValueError(f'Invalid computing rate for device {d}')
        if any(row[name] < 0 for name in row if name.endswith('_bytes')):
            raise ValueError('Negative device memory parameter')
        if row['M_OTA_bytes'] <= 0 or row['M_SRAM_bytes'] <= 0:
            raise ValueError('Nonpositive memory capacity')
        if row['arena_memory'] not in ('SRAM', 'PSRAM'):
            raise ValueError('arena_memory must be SRAM or PSRAM')
    primaries = []
    for k, row in segments.items():
        if row['primary_device_id'] not in devices or row['omega_k'] <= 0:
            raise ValueError('Invalid primary device or operator count')
        primaries.append(row['primary_device_id'])
        if any(row[name] < 0 for name in row if name.endswith('_bytes')):
            raise ValueError('Negative segment memory/activation parameter')
        if row['omega_k'] != row['operator_end_exclusive'] - row['operator_start_inclusive']:
            raise ValueError('omega_k differs from the converted-operator range')
        if k < len(segments):
            nxt = segments[k + 1]
            if row['operator_end_exclusive'] != nxt['operator_start_inclusive'] or row['v_next_bytes'] != nxt['v_k_bytes']:
                raise ValueError('Segment boundaries or activation volumes do not match')
    if len(primaries) != len(set(primaries)):
        raise ValueError('This paper-aligned input requires at most one initial primary per device')
    if segments[1]['operator_start_inclusive'] != 0 or sum(s['omega_k'] for s in segments.values()) != case['omega_Theta']:
        raise ValueError('Segments must cover all converted operators and sum to omega_Theta')
    if segments[1]['primary_device_id'] != case['source_device_id']:
        raise ValueError('The first segment must be on the source device')
    for (i, j, p), rate in links.items():
        if i not in devices or j not in devices or i == j or rate <= 0 or not math.isfinite(rate):
            raise ValueError('Invalid directed throughput entry')
        if any(p not in devices[d]['supported_protocols'].split(';') for d in (i, j)):
            raise ValueError('Link protocol is incompatible with an endpoint')
        if p == 'ESP-NOW' and any(not devices[d]['device_type'].startswith('ESP32') for d in (i, j)):
            raise ValueError('ESP-NOW is restricted to ESP32 pairs in this setup')


def compute_seconds(segment, device, case):
    # Paper Eqs. (10)-(11). Do not replace with PC measurements.
    return segment['v_k_bytes'] / device['phi_bytes_per_s'] * segment['omega_k'] / case['omega_Theta']


def memory_parts(device, segment, case):
    """Per-segment SRAM and PSRAM charges, kept in separate constraints."""
    u = segment['u_k_bytes']
    if case['memory_policy'] == 'paper_sram':
        return u, 0  # Literal Eq. (15d); no extra per-segment extension.
    outside = device['U_extra_per_segment_bytes']
    return (outside, u) if device['arena_memory'] == 'PSRAM' else (outside + u, 0)


def link_options(links, i, j, case):
    if i == j:
        return [('LOCAL', math.inf)]
    selected = case['protocol_policy']
    return sorted((p, b) for (a, c, p), b in links.items()
                  if (a, c) == (i, j) and (selected == 'joint' or p == selected))


def check_placement(devices, segments, links, case, placement, protocols=None, partial=False):
    """Recompute original constraints and latency, independent of solver rows."""
    failed = set(case['failed_device_ids'])
    survivors = set(devices) - failed
    errors = []
    if (not set(placement) <= set(segments) or
        (not partial and set(placement) != set(segments)) or
        any(d not in survivors for d in placement.values())):
        return dict(feasible=False, errors=['Incomplete placement or a host is failed/unknown'])
    for k, s in segments.items():
        if s['primary_device_id'] not in failed and placement.get(k) != s['primary_device_id']:
            errors.append(f'Surviving primary S{k} was moved')
    resources = []
    for d in sorted(survivors):
        n = devices[d]
        hosted = sorted(k for k, host in placement.items() if host == d)
        primary = [k for k in hosted if segments[k]['primary_device_id'] == d]
        replicas = [k for k in hosted if k not in primary]
        model_bytes = sum(segments[k]['m_k_bytes'] for k in hosted)
        flash = n['M_base_bytes'] + model_bytes
        sram = n['U_sys_bytes'] + n['U_comm_bytes']
        psram = n['U_psram_reserved_bytes'] if case['memory_policy'] == 'esp32_psram' else 0
        for k in hosted:
            a, b = memory_parts(n, segments[k], case)
            sram += a
            psram += b
        if flash > n['M_OTA_bytes']:
            errors.append(f'Device {d} exceeds its OTA Flash slot')
        if model_bytes > n['M_model_budget_bytes']:
            errors.append(f'Device {d} exceeds its model-storage budget')
        if sram > n['M_SRAM_bytes']:
            errors.append(f'Device {d} exceeds its SRAM capacity')
        if psram > n['M_PSRAM_bytes']:
            errors.append(f'Device {d} exceeds its PSRAM capacity')
        resources.append(dict(device_id=d, primary_segments=primary, replica_segments=replicas,
                              model_used_bytes=model_bytes, model_limit_bytes=n['M_model_budget_bytes'],
                              flash_used_bytes=flash, flash_limit_bytes=n['M_OTA_bytes'],
                              sram_used_bytes=sram, sram_limit_bytes=n['M_SRAM_bytes'],
                              psram_used_bytes=psram, psram_limit_bytes=n['M_PSRAM_bytes']))
    transfers = []
    comm = 0.0
    for k in range(1, len(segments)):
        if k not in placement or k + 1 not in placement:
            continue
        i, j = placement[k], placement[k + 1]
        options = dict(link_options(links, i, j, case))
        if not options:
            errors.append(f'No compatible selected-protocol link at boundary {k}')
            continue
        p = protocols.get(k) if protocols is not None else max(options, key=lambda p: options[p])
        if p not in options:
            errors.append(f'Invalid protocol at boundary {k}')
            continue
        delay = segments[k]['v_next_bytes'] / options[p]
        comm += delay
        transfers.append(dict(boundary=k, source_id=i, destination_id=j, protocol=p,
                              activation_bytes=segments[k]['v_next_bytes'], seconds=delay))
    comp = sum(compute_seconds(segments[k], devices[d], case) for k, d in placement.items())
    return dict(feasible=not errors, complete=set(placement) == set(segments), errors=errors, placement=placement, transfers=transfers,
                resources=resources, computation_seconds=comp, communication_seconds=comm,
                objective_seconds=comp + comm)


class LinearModel:
    """Small sparse MILP builder. An expression is (coefficient_dict, constant)."""
    def __init__(self):
        self.names, self.integer, self.costs = [], [], []
        self.rows, self.lows, self.highs, self.row_names = [], [], [], []
        self.objective_constant = 0.0

    def variable(self, name, integer):
        index = len(self.names)
        self.names.append(name)
        self.integer.append(int(integer))
        self.costs.append(0.0)
        return ({index: 1.0}, 0.0)

    def constraint(self, name, expr, lower, upper):
        self.row_names.append(name)
        self.rows.append(expr[0])
        self.lows.append(lower - expr[1])
        self.highs.append(upper - expr[1])

    def cost(self, expr, coefficient):
        for index, value in expr[0].items():
            self.costs[index] += coefficient * value
        self.objective_constant += coefficient * expr[1]


def combine(*terms):
    # Each term: multiplier, expression.
    result, constant = {}, 0.0
    for multiplier, (coefficients, offset) in terms:
        constant += multiplier * offset
        for index, value in coefficients.items():
            result[index] = result.get(index, 0.0) + multiplier * value
    return {i: v for i, v in result.items() if v}, constant


def build_model(devices, segments, links, case, fixed_placement=None):
    model = LinearModel()
    failed = set(case['failed_device_ids'])
    survivors = sorted(set(devices) - failed)
    failed_segments = {k for k, s in segments.items() if s['primary_device_id'] in failed}
    # Eq. (17): fixed surviving primary, or any surviving host for a replica.
    H = {k: survivors if k in failed_segments else [s['primary_device_id']] for k, s in segments.items()}
    x, h, y, z = {}, {}, {}, {}
    for k in segments:
        for d in H[k]:
            if k in failed_segments:
                x[k, d] = model.variable(f'x[{k},{d}]', integer=True)  # Eq. (15e)
                h[k, d] = x[k, d]
            else:
                h[k, d] = ({}, 1.0)  # Eq. (18): constant, not a new variable.
            model.cost(h[k, d], compute_seconds(segments[k], devices[d], case))
        if k in failed_segments:
            model.constraint(f'15b_one_replica[{k}]', combine(*[(1, x[k, d]) for d in H[k]]), 1, 1)
            if fixed_placement is not None:
                for d in H[k]:
                    selected = int(fixed_placement[k] == d)
                    model.constraint(f'fixed_heuristic_host[{k},{d}]', x[k, d], selected, selected)

    for d in survivors:
        n = devices[d]
        fixed = [k for k in segments if k not in failed_segments and segments[k]['primary_device_id'] == d]
        primary_flash = sum(segments[k]['m_k_bytes'] for k in fixed)
        # Eq. (15c), scaled to KiB for the numerical solver.
        expr = combine(*[(segments[k]['m_k_bytes'] / 1024, x[k, d]) for k in sorted(failed_segments)])
        model.constraint(f'15c_flash[{d}]', expr, -math.inf, (n['M_OTA_bytes'] - n['M_base_bytes'] - primary_flash) / 1024)
        model.constraint(f'model_storage_budget[{d}]', expr, -math.inf,
                         (n['M_model_budget_bytes'] - primary_flash) / 1024)
        fixed_sram = sum(memory_parts(n, segments[k], case)[0] for k in fixed)
        fixed_psram = sum(memory_parts(n, segments[k], case)[1] for k in fixed)
        expr = combine(*[(memory_parts(n, segments[k], case)[0] / 1024, x[k, d]) for k in sorted(failed_segments)])
        model.constraint(f'15d_sram[{d}]', expr, -math.inf, (n['M_SRAM_bytes'] - n['U_sys_bytes'] - n['U_comm_bytes'] - fixed_sram) / 1024)
        if case['memory_policy'] == 'esp32_psram':
            expr = combine(*[(memory_parts(n, segments[k], case)[1] / 1024, x[k, d]) for k in sorted(failed_segments)])
            model.constraint(f'extension_psram[{d}]', expr, -math.inf, (n['M_PSRAM_bytes'] - n['U_psram_reserved_bytes'] - fixed_psram) / 1024)

    for k in range(1, len(segments)):
        for i in H[k]:
            for j in H[k + 1]:
                left, right = h[k, i], h[k + 1, j]
                if k in failed_segments and k + 1 in failed_segments:
                    pair = model.variable(f'y[{k},{i},{j}]', integer=False)
                    y[k, i, j] = pair  # Bounds 0 <= y <= 1.
                    # The three inequalities in Eq. (19), explicitly implemented.
                    model.constraint(f'19a_y_le_left[{k},{i},{j}]', combine((1, pair), (-1, left)), -math.inf, 0)
                    model.constraint(f'19b_y_le_right[{k},{i},{j}]', combine((1, pair), (-1, right)), -math.inf, 0)
                    model.constraint(f'19c_y_ge_sum_minus_1[{k},{i},{j}]', combine((1, pair), (-1, left), (-1, right)), -1, math.inf)
                elif k in failed_segments:
                    pair = left  # x * 1 = x
                elif k + 1 in failed_segments:
                    pair = right  # 1 * x = x
                else:
                    pair = ({}, 1.0)
                if i == j:
                    continue  # Eq. (13): local transfer has zero cost.
                options = link_options(links, i, j, case)
                if not options:
                    # A missing link is forbidden, not represented by an invented rate.
                    model.constraint(f'forbid_unavailable_link[{k},{i},{j}]', pair, 0, 0)
                elif case['protocol_policy'] != 'joint':
                    model.cost(pair, segments[k]['v_next_bytes'] / options[0][1])
                else:
                    # Revised paper Eq. (20): choose one p for the selected pair.
                    choices = []
                    for p, rate in options:
                        z[k, i, j, p] = model.variable(f'z[{k},{i},{j},{p}]', integer=True)
                        choices.append((1, z[k, i, j, p]))
                        model.cost(z[k, i, j, p], segments[k]['v_next_bytes'] / rate)
                    model.constraint(f'20_protocol_sum[{k},{i},{j}]', combine(*choices, (-1, pair)), 0, 0)
    return model, x, y, z


def solve(devices, segments, links, case, fixed_placement=None):
    start = time.perf_counter()
    if fixed_placement is not None:
        checked = check_placement(devices, segments, links, case, fixed_placement)
        if not checked['feasible']:
            raise ValueError('Invalid fixed heuristic placement: ' + '; '.join(checked['errors']))
    model, x, y, z = build_model(devices, segments, links, case, fixed_placement)
    build_seconds = time.perf_counter() - start
    if model.names:
        r, c, values = [], [], []
        for row, coefficients in enumerate(model.rows):
            for col, value in coefficients.items():
                r.append(row); c.append(col); values.append(value)
        matrix = coo_matrix((values, (r, c)), shape=(len(model.rows), len(model.names))).tocsc()
        result = milp(c=np.array(model.costs), integrality=np.array(model.integer),
                      bounds=Bounds(np.zeros(len(model.names)), np.ones(len(model.names))),
                      constraints=LinearConstraint(matrix, model.lows, model.highs),
                      options=case['solver'])
        status = {0: 'optimal', 1: 'limit_reached', 2: 'proven_infeasible', 3: 'unbounded', 4: 'solver_error'}.get(result.status, 'solver_error')
        values = result.x
        raw_bound = getattr(result, 'mip_dual_bound', None)
        raw_gap = getattr(result, 'mip_gap', None)
        message = result.message
    else:
        # No failed segments + fixed p: the model is a constant feasibility check.
        good = all(low <= 0 <= high for low, high in zip(model.lows, model.highs))
        status = 'optimal' if good else 'proven_infeasible'
        values = np.array([]) if good else None
        raw_bound, raw_gap = (0.0, 0.0) if good else (None, None)
        message = 'Constant model evaluated directly; no decision variables'
    def finite(value):
        return float(value) if value is not None and math.isfinite(value) else None
    report = dict(status=status, message=message, failed_device_ids=case['failed_device_ids'],
                  optimization_scope='protocols_for_fixed_placement' if fixed_placement is not None else 'joint_placement_protocols',
                  protocol_policy=case['protocol_policy'], memory_policy=case['memory_policy'],
                  variable_counts=dict(x_binary=len(x), y_continuous=len(y), z_binary=len(z)),
                  constraint_count=len(model.rows), objective_constant_seconds=model.objective_constant,
                  scipy_version=scipy.__version__, numpy_version=np.__version__,
                  solver_reported_relative_gap=finite(raw_gap),
                  build_seconds=build_seconds, feasible=False)
    if values is not None:
        integers = np.array(model.integer, dtype=bool)
        if np.any(np.abs(values[integers] - np.rint(values[integers])) > 1e-5):
            raise RuntimeError('Solver incumbent is not integral')
        def value(expr):
            return sum(values[i] * v for i, v in expr[0].items()) + expr[1]
        failed = set(case['failed_device_ids'])
        placement = {k: s['primary_device_id'] for k, s in segments.items() if s['primary_device_id'] not in failed}
        for k in segments:
            if k not in placement:
                hosts = [d for (kk, d), expr in x.items() if kk == k and value(expr) > .5]
                if len(hosts) != 1:
                    raise RuntimeError(f'Segment {k} does not have exactly one selected replica host')
                placement[k] = hosts[0]
        protocols = {}
        for k in range(1, len(segments)):
            i, j = placement[k], placement[k + 1]
            if i == j:
                protocols[k] = 'LOCAL'
            elif case['protocol_policy'] == 'joint':
                selected = [p for (kk, ii, jj, p), expr in z.items() if (kk, ii, jj) == (k, i, j) and value(expr) > .5]
                if len(selected) != 1:
                    raise RuntimeError('No unique selected protocol')
                protocols[k] = selected[0]
            else:
                protocols[k] = case['protocol_policy']
        for (k, i, j), expr in y.items():
            expected = int(placement[k] == i and placement[k + 1] == j)
            if abs(value(expr) - expected) > 1e-6:
                raise RuntimeError('Eq. (19) product verification failed')
        checked = check_placement(devices, segments, links, case, placement, protocols)
        if not checked['feasible']:
            raise RuntimeError('Independent verification failed: ' + '; '.join(checked['errors']))
        linear_objective = float(np.dot(model.costs, values))
        full_objective = model.objective_constant + linear_objective
        if not math.isclose(full_objective, checked['objective_seconds'], rel_tol=1e-7, abs_tol=1e-8):
            raise RuntimeError('MILP objective does not match independent calculation')
        report.update(checked)
        report['independent_verification_passed'] = True
        bound = finite(raw_bound)
        if status == 'optimal' and bound is None:
            bound = linear_objective
        report['objective_bound_seconds'] = bound + model.objective_constant if bound is not None else None
        report['full_objective_relative_gap'] = (abs(full_objective - report['objective_bound_seconds']) / max(abs(full_objective), 1e-12)) if bound is not None else None
        report['variables'] = [{'name': name, 'value': float(v)} for name, v in zip(model.names, values)]
    report['algorithm_seconds'] = time.perf_counter() - start
    return report, model


def enumerate_reference(devices, segments, links, case):
    """Small-case oracle: original objective and constraints, without x/y/z."""
    failed = set(case['failed_device_ids'])
    survivors = sorted(set(devices) - failed)
    fixed = {k: s['primary_device_id'] for k, s in segments.items() if s['primary_device_id'] not in failed}
    missing = sorted(set(segments) - set(fixed))
    combinations = len(survivors) ** len(missing)
    if combinations > 200000:
        raise ValueError('Enumeration is restricted to 200000 placements; omit --verify for larger cases')
    best, count = math.inf, 0
    for selected in itertools.product(survivors, repeat=len(missing)):
        placement = {**fixed, **dict(zip(missing, selected))}
        checked = check_placement(devices, segments, links, case, placement)
        if checked['feasible']:
            count += 1
            best = min(best, checked['objective_seconds'])
    return dict(placements_checked=combinations, feasible_placements=count,
                best_objective_seconds=best if count else None)


def save_report(output, report, model, devices, segments, input_dir, case, audit=True):
    output.mkdir(parents=True, exist_ok=True)
    report['case_used'] = case
    report['input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(input_dir.iterdir()) if p.suffix in ('.csv', '.json')}
    if not audit:
        report.pop('variables', None)  # Large-scale runs retain placements/transfers, not every x/y/z value.
    (output / 'solution.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not audit:
        return
    # Audit view: labels follow the paper equation numbering and every row is visible.
    rows = [dict(name=name, lower=str(low), upper=str(high),
                 coefficients=json.dumps({model.names[i]: v for i, v in row.items()}, separators=(',', ':')))
            for name, row, low, high in zip(model.row_names, model.rows, model.lows, model.highs)]
    write_csv(output / 'constraints.csv', rows, ['name', 'lower', 'upper', 'coefficients'])
    if report['feasible']:
        placement = report['placement']
        write_csv(output / 'placement.csv', [dict(segment_id=k, original_device_id=segments[k]['primary_device_id'],
                  selected_device_id=placement[k], role='fixed_primary' if placement[k] == segments[k]['primary_device_id'] else 'replica',
                  computation_seconds=compute_seconds(segments[k], devices[placement[k]], case)) for k in sorted(segments)])
        write_csv(output / 'transfers.csv', report['transfers'], ['boundary','source_id','destination_id','protocol','activation_bytes','seconds'])
        write_csv(output / 'resources.csv', report['resources'])
        write_csv(output / 'variables.csv', report['variables'], ['name', 'value'])
    else:
        # Prevent a failed new solve leaving successful-looking stale files.
        for name in ('placement.csv', 'transfers.csv', 'resources.csv', 'variables.csv'):
            (output / name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'inputs')
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    parser.add_argument('--failed', help='Comma-separated IDs, e.g. 3,4; use none for no failures')
    parser.add_argument('--protocol', choices=['BLE','UDP','TCP','ESP-NOW','joint'], default='joint')
    parser.add_argument('--memory', choices=['esp32_psram','paper_sram'])
    parser.add_argument('--esp32-model-mib', help='ESP32 model quota in MiB; default 1.3 unless case.json overrides it')
    parser.add_argument('--verify', action='store_true', help='Compare this small case against exhaustive host enumeration')
    args = parser.parse_args()
    try:
        failed = None if args.failed is None else ([] if args.failed.lower() == 'none' else [int(i) for i in args.failed.split(',')])
        devices, segments, links, case = load_inputs(args.inputs, failed, args.protocol, args.memory, args.esp32_model_mib)
        initial_case = {**case, 'failed_device_ids': []}
        initial = check_placement(devices, segments, links, initial_case,
                                  {k: s['primary_device_id'] for k, s in segments.items()})
        if not initial['feasible']:
            print('NOTE: Initial deployment under this policy is infeasible: ' + '; '.join(initial['errors']))
        report, model = solve(devices, segments, links, case)
        report.update(experiment_metadata(devices, segments, links, case))
        report['initial_deployment_feasible'] = initial['feasible']
        report['initial_deployment_errors'] = initial['errors']
        if args.verify:
            reference = enumerate_reference(devices, segments, links, case)
            if report['status'] == 'optimal':
                if not reference['feasible_placements'] or not math.isclose(report['objective_seconds'], reference['best_objective_seconds'], rel_tol=1e-7, abs_tol=1e-8):
                    raise RuntimeError('MILP optimum disagrees with exhaustive enumeration')
                reference['matches_milp_optimum'] = True
            elif report['status'] == 'proven_infeasible':
                if reference['feasible_placements']:
                    raise RuntimeError('MILP infeasibility disagrees with exhaustive enumeration')
                reference['matches_milp_infeasibility'] = True
            else:
                reference['verification_note'] = 'MILP did not finish with an optimal/infeasible proof; no match claim'
            report['enumeration'] = reference
        save_report(args.output, report, model, devices, segments, args.inputs, case)
        print('Status:', report['status'])
        print('Failed devices:', case['failed_device_ids'], '| Protocol:', case['protocol_policy'], '| Memory:', case['memory_policy'])
        print('Model budgets (bytes):', report['effective_model_budgets_bytes'])
        print('Variables:', report['variable_counts'], '| Constraints:', report['constraint_count'])
        if report['feasible']:
            for k in sorted(segments):
                host = report['placement'][k]
                role = 'fixed primary' if host == segments[k]['primary_device_id'] else 'replica'
                print(f'  S{k} -> device {host} ({role})')
            for tr in report['transfers']:
                print(f"  S{tr['boundary']} -> S{tr['boundary']+1}: {tr['protocol']}, {tr['seconds']*1000:.3f} ms")
            print(f"Computation: {report['computation_seconds']:.6f} s")
            print(f"Communication: {report['communication_seconds']:.6f} s")
            print(f"E2E objective: {report['objective_seconds']:.6f} s")
            print('Independent memory, placement, link and objective checks: PASS')
        if args.verify:
            print('Enumeration:', report['enumeration'])
        print('Saved:', args.output.resolve())
        print('Input rates and runtime overheads are assumptions; no measured MCU timing claim.')
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f'MILP run failed: {exc}\n')


if __name__ == '__main__':
    main()
