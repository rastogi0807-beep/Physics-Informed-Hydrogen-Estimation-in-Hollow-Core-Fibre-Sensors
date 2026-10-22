"""Execute the prespecified, synthetic hydrogen concentration benchmark."""
import time
import csv
import json
import sys
import platform
import itertools
import importlib.metadata
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.stats import chi2
from scipy.optimize import brentq
from physics import *

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results'
R.mkdir(exist_ok=True)
METHODS = ['Nominal environment', 'Plug-in temperature pressure', 'Joint uncertainty', 'In-phase only', 'Single frequency equal time']
STRATA = [(0.002, .05), (.05, .30), (.30, .65), (.65, .95)]


def write_csv(name, records):
    if not records:
        return
    with (R / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def sensitivity(theta, frequencies):
    scales = np.array([1., 10., 20000., .1, .2, .01])
    steps = np.array([1e-5, .01, 10., 1e-4, 1e-4, 1e-5])
    inv = Inverse(frequencies)
    columns = []
    for j in range(6):
        plus = np.array(theta, dtype=float)
        minus = plus.copy()
        plus[j] += steps[j]
        minus[j] -= steps[j]
        d = inv.project((forward(plus, frequencies) - forward(minus, frequencies)) / (2 * steps[j]))
        columns.append(np.r_[d.real.ravel(), d.imag.ravel()] * scales[j])
    J = np.array(columns).T
    prior = np.zeros((4, 6))
    for row, (col, std) in enumerate([(1, C.sigma_T), (2, C.sigma_p), (3, C.sigma_log_gain), (4, C.sigma_log_relaxation)]):
        prior[row, col] = scales[col] / std
    F = J.T @ J + prior.T @ prior
    cov = np.linalg.pinv(F, rcond=1e-12)
    sv = np.linalg.svd(J, compute_uv=False)
    return float(np.sqrt(cov[0, 0])), float(sv[-1] / sv[0]), F, scales


def select_design():
    grid = [np.array([x, T, p, 0, 0, 0]) for x in [.02, .30, .85] for T in [280, 300, 325] for p in [90000, 150000]]
    records = []
    for triple in itertools.combinations(CANDIDATES, 3):
        values = [sensitivity(theta, triple)[0] for theta in grid]
        records.append({'f1_Hz': triple[0], 'f2_Hz': triple[1], 'f3_Hz': triple[2], 'worst_std_x': max(values), 'median_std_x': np.median(values), 'design_cases': len(grid)})
    write_csv('frequency_design.csv', records)
    best = min(records, key=lambda row: row['worst_std_x'])
    freq = np.array([best[key] for key in ['f1_Hz', 'f2_Hz', 'f3_Hz']])
    print('Selected modulation frequencies', freq.tolist(), flush=True)
    return freq


def generate(name, n_per_stratum, frequencies, seed):
    rng = np.random.default_rng(seed)
    theta, aux, obs, single, records = [], [], [], [], []
    for stratum, (low, high) in enumerate(STRATA):
        for _ in range(n_per_stratum):
            t = np.array([rng.uniform(low, high), rng.uniform(270, 330), rng.uniform(80000, 160000), rng.normal(0, .05), rng.normal(0, .12), rng.normal(0, .006)])
            theta.append(t)
            auxiliary = auxiliary_measurement(t, rng)
            aux.append(auxiliary)
            obs.append(observation(t, frequencies, rng))
            single.append(observation(t, frequencies[:1], rng, sigma=C.noise_rad / np.sqrt(3)))
            records.append({'id': len(records), 'stratum': stratum, 'x': t[0], 'T_K': t[1], 'p_Pa': t[2], 'log_gain': t[3], 'log_relaxation': t[4], 'shift_cm-1': t[5], 'T_observed_K': auxiliary[0], 'p_observed_Pa': auxiliary[1], 'log_gain_observed': auxiliary[2], 'log_relaxation_observed': auxiliary[3]})
    theta, aux, obs, single = map(np.asarray, (theta, aux, obs, single))
    np.savez_compressed(R / (name + '_data.npz'), theta=theta, auxiliary=aux, complex_phase=obs, single_equal_time=single, frequencies_Hz=frequencies, detuning_cm1=DETUNING)
    write_csv(name + '_configurations.csv', records)
    return theta, aux, obs, single


def fit_method(method, theta, aux, obs, single, frequencies, tag):
    mode = {'Nominal environment': 'nominal', 'Plug-in temperature pressure': 'plugin', 'Joint uncertainty': 'joint', 'In-phase only': 'inphase', 'Single frequency equal time': 'joint'}[method]
    is_single = method.startswith('Single')
    inv = Inverse(frequencies[:1] if is_single else frequencies, mode=mode, sigma=C.noise_rad / np.sqrt(3) if is_single else C.noise_rad)
    records = []
    for j, (t, a, y, one) in enumerate(zip(theta, aux, obs, single)):
        start = time.perf_counter()
        result = inv.fit(one if is_single else y, a)
        estimated = result['theta']
        records.append({'id': j, 'method': method, 'stratum': j // (len(theta) // 4), 'true_x': t[0], 'estimated_x': estimated[0], 'standard_error_x': result['se_x'], 'T_estimated_K': estimated[1], 'p_estimated_Pa': estimated[2], 'log_gain_estimated': estimated[3], 'log_relaxation_estimated': estimated[4], 'shift_estimated_cm-1': estimated[5], 'red_residual': result['red_residual'], 'cost': result['cost'], 'branch_gap': result['branch_gap'], 'optimizer_success': int(result['success']), 'time_s': time.perf_counter() - start})
        if (j + 1) % 80 == 0:
            print(tag, method, j + 1, '/', len(theta), flush=True)
    return records


def calibration(records):
    scales = {}
    thresholds = {}
    for method in METHODS:
        subset = [row for row in records if row['method'] == method]
        standardized = [abs(row['estimated_x'] - row['true_x']) / max(row['standard_error_x'], 1e-12) for row in subset]
        scales[method] = max(1.96, float(np.quantile(standardized, .95, method='higher')))
        thresholds[method] = float(np.quantile([row['red_residual'] for row in subset], .99, method='higher'))
    return scales, thresholds


def annotate(records, scales, thresholds):
    for row in records:
        half = scales[row['method']] * row['standard_error_x']
        lower, upper = max(0, row['estimated_x'] - half), min(.98, row['estimated_x'] + half)
        row.update({'lower_x': lower, 'upper_x': upper, 'covered': int(lower <= row['true_x'] <= upper), 'raw_covered': int(abs(row['true_x'] - row['estimated_x']) <= 1.96 * row['standard_error_x']), 'accepted': int(row['red_residual'] <= thresholds[row['method']] and row['branch_gap'] > chi2.ppf(.95, 1) and upper - lower <= .15)})
    return records


def metrics(records):
    err = np.array([row['estimated_x'] - row['true_x'] for row in records])
    accepted = np.array([row['accepted'] for row in records], dtype=bool)
    return {'n': len(records), 'mae_pp': float(np.mean(abs(err)) * 100), 'rmse_pp': float(np.sqrt(np.mean(err**2)) * 100), 'bias_pp': float(np.mean(err) * 100), 'p95_error_pp': float(np.quantile(abs(err), .95) * 100), 'within_5pp': float(np.mean(abs(err) <= .05)), 'coverage': float(np.mean([row['covered'] for row in records])), 'raw_coverage': float(np.mean([row['raw_covered'] for row in records])), 'median_width_pp': float(np.median([row['upper_x'] - row['lower_x'] for row in records]) * 100), 'accepted': int(accepted.sum()), 'accepted_fraction': float(accepted.mean()), 'accepted_mae_pp': float(np.mean(abs(err[accepted])) * 100) if accepted.any() else None, 'wrong_accepted_5pp': int(np.sum(accepted & (abs(err) > .05))), 'ambiguous_multistart': int(sum(row['branch_gap'] <= chi2.ppf(.95, 1) for row in records)), 'optimizer_failed': int(sum(not row['optimizer_success'] for row in records))}


def verification(frequencies):
    theta = np.array([.30, 300, 100000, 0, 0, 0])
    k, cp, refr, tau = gas_properties(*theta[:3])
    reference = fv_thermal(k, cp, frequencies, cells=1920)
    records = []
    for cells in [30, 60, 120, 240, 480, 960]:
        value = fv_thermal(k, cp, frequencies, cells=cells)
        records.append({'check': 'Finite volume radial refinement', 'resolution': cells, 'relative_error': float(np.linalg.norm(value - reference) / np.linalg.norm(reference))})
    for modes in [4, 8, 16, 40, 80]:
        value = ModalThermal(modes).transfer(k, cp, frequencies)
        records.append({'check': 'Bessel modal truncation', 'resolution': modes, 'relative_error': float(np.linalg.norm(value - reference) / np.linalg.norm(reference))})
    # Uniform volumetric heating q=1 has the exact steady solution (R^2-r^2)/(4k).
    modal = ModalThermal(80)
    r = np.linspace(0, C.radius, 1001)
    zeta = modal.beta
    source = 2 / (zeta * j1(zeta))
    solution = np.sum((source / (k * (zeta / C.radius)**2))[:, None] * j0(zeta[:, None] * r[None] / C.radius), axis=0)
    analytical = (C.radius**2 - r**2) / (4 * k)
    uniform_error = float(np.linalg.norm(solution - analytical) / np.linalg.norm(analytical))
    write_csv('numerical_verification.csv', records)
    _, r, field = fv_thermal(k, cp, frequencies, return_field=True)
    alpha = spectroscopic(*theta[:3])[24]
    fields_K = field * alpha * C.pump_power / (1 + 1j * 2 * np.pi * frequencies[:, None] * tau)
    np.savez_compressed(R / 'illustrative_thermal_field.npz', radius_m=r, complex_temperature_K=fields_K, frequencies_Hz=frequencies, theta=theta)
    return {'uniform_heating_relative_L2_error': uniform_error, 'illustrative_peak_temperature_K': float(np.max(abs(fields_K))), 'max_illustrative_optical_depth': float(np.max(spectroscopic(*theta[:3])) * C.length)}


def bootstrap(test_records):
    methods = {m: [row for row in test_records if row['method'] == m] for m in METHODS}
    groups = [np.arange(i * 80, (i + 1) * 80) for i in range(4)]
    rng = np.random.default_rng(C.seed + 600)
    distributions = {m: [] for m in METHODS}
    for _ in range(2000):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for method in METHODS:
            err = np.array([abs(row['estimated_x'] - row['true_x']) for row in methods[method]])
            distributions[method].append(float(np.mean(err[idx]) * 100))
    out = {method: {'mae_pp_ci95': np.quantile(values, [.025, .975]).tolist()} for method, values in distributions.items()}
    for method in ['Plug-in temperature pressure', 'Single frequency equal time', 'In-phase only']:
        d = np.array(distributions[method]) - np.array(distributions['Joint uncertainty'])
        out['paired_mae_improvement_vs_' + method] = {'mean_pp': float(metrics(methods[method])['mae_pp'] - metrics(methods['Joint uncertainty'])['mae_pp']), 'ci95_pp': np.quantile(d, [.025, .975]).tolist()}
    (R / 'bootstrap_summary.json').write_text(json.dumps(out, indent=2))
    write_csv('bootstrap_replicates.csv', [{'replicate': j, **{method: distributions[method][j] for method in METHODS}} for j in range(2000)])
    return out


def main():
    start = time.perf_counter()
    frequencies = select_design()
    verify = verification(frequencies)
    val = generate('calibration', 30, frequencies, C.seed + 1)
    test = generate('test', 80, frequencies, C.seed + 2)
    val_records = []
    for method in METHODS:
        val_records += fit_method(method, *val, frequencies, 'Calibration')
        write_csv('calibration_estimates.csv', val_records)
    scales, thresholds = calibration(val_records)
    annotate(val_records, scales, thresholds)
    write_csv('calibration_estimates.csv', val_records)
    test_records = []
    for method in METHODS:
        records = fit_method(method, *test, frequencies, 'Test')
        annotate(records, scales, thresholds)
        test_records += records
        write_csv('test_estimates.csv', test_records)
        print(method, metrics(records), flush=True)
    overall = {method: metrics([row for row in test_records if row['method'] == method]) for method in METHODS}
    stratified = {method: {str(s): metrics([row for row in test_records if row['method'] == method and row['stratum'] == s]) for s in range(4)} for method in METHODS}
    boot = bootstrap(test_records)
    # Scalar low-branch peak calibration: a diagnostic control, not a competitive sole baseline.
    xgrid = np.linspace(.000001, .98, 1001)
    curve = np.array([abs(forward([x, 300, 100000, 0, 0, 0], frequencies[:1])[0, 24]) for x in xgrid])
    turning = int(curve.argmax())
    scalar = []
    for j, (truth, data) in enumerate(zip(test[0], test[2])):
        amp = abs(data[0, 24])
        est = float(np.interp(amp, curve[:turning + 1], xgrid[:turning + 1]))
        scalar.append({'id': j, 'true_x': truth[0], 'estimated_x': est, 'peak_phase_rad': amp})
    write_csv('scalar_peak_control.csv', scalar)
    # Same 64 test configurations used in all nuisance/challenge scenarios.
    idx = np.concatenate([np.arange(s * 80, s * 80 + 16) for s in range(4)])
    challenge = []
    challenge_arrays = {}
    for scenario, options in [('Large auxiliary uncertainty', {'sigma_T': 8., 'sigma_p': 18000.}), ('Biased auxiliary measurements', {'bias_T': 6., 'bias_p': 12000.}), ('Thermal spectral mismatch', {'extra': {'k_factor': 1.15, 'broadening_factor': 1.08, 'pump_fraction': .70, 'extra_pole': .5e-6}}), ('Unmodelled lock-in phase', {'extra': {'phase_error': np.deg2rad(4)}})]:
        rng = np.random.default_rng(C.seed + 100 + len(challenge_arrays))
        ys, auxs = [], []
        sigma_T, sigma_p = options.get('sigma_T', C.sigma_T), options.get('sigma_p', C.sigma_p)
        for j in idx:
            truth = test[0][j]
            ys.append(observation(truth, frequencies, rng, extra=options.get('extra')))
            a = auxiliary_measurement(truth, rng, sigma_T, sigma_p)
            a[:2] += [options.get('bias_T', 0), options.get('bias_p', 0)]
            auxs.append(a)
        challenge_arrays[scenario] = (np.array(ys), np.array(auxs))
        inv = Inverse(frequencies, sigma_T=sigma_T, sigma_p=sigma_p)
        records = []
        for j, y, a in zip(idx, ys, auxs):
            result = inv.fit(y, a)
            records.append({'scenario': scenario, 'source_id': int(j), 'id': len(records), 'method': 'Joint uncertainty', 'stratum': int(j // 80), 'true_x': test[0][j, 0], 'estimated_x': result['theta'][0], 'standard_error_x': result['se_x'], 'T_estimated_K': result['theta'][1], 'p_estimated_Pa': result['theta'][2], 'log_gain_estimated': result['theta'][3], 'log_relaxation_estimated': result['theta'][4], 'shift_estimated_cm-1': result['theta'][5], 'red_residual': result['red_residual'], 'cost': result['cost'], 'branch_gap': result['branch_gap'], 'optimizer_success': int(result['success']), 'time_s': 0.})
        annotate(records, scales, thresholds)
        challenge += records
        print('Challenge', scenario, metrics(records), flush=True)
    write_csv('challenge_estimates.csv', challenge)
    np.savez_compressed(R / 'challenge_data.npz', source_test_ids=idx, **{f'phase_{k}': value[0] for k, value in enumerate(challenge_arrays.values())}, **{f'auxiliary_{k}': value[1] for k, value in enumerate(challenge_arrays.values())})
    (R / 'challenge_scenarios.json').write_text(json.dumps({str(k): key for k, key in enumerate(challenge_arrays)}, indent=2))
    # Profile scans audit global concentration ambiguity on prespecified cases.
    profiles = []
    inv = Inverse(frequencies)
    chosen = [5, 35, 85, 115, 165, 195, 245, 275]
    for j in chosen:
        best = next(row for row in test_records if row['method'] == 'Joint uncertainty' and row['id'] == j)
        scan = np.unique(np.r_[np.linspace(.0001, .98, 121), np.linspace(max(.0001, best['estimated_x'] - .15), min(.98, best['estimated_x'] + .15), 81), best['estimated_x'], test[0][j, 0]])
        for x in scan:
            result = inv.fit(test[2][j], test[1][j], fixed_x=float(x))
            profiles.append({'test_id': j, 'true_x': test[0][j, 0], 'x': float(x), 'profile_cost': result['cost'], 'delta_cost': max(0, result['cost'] - best['cost'])})
        print('Profile scan complete', j, flush=True)
    write_csv('concentration_profiles.csv', profiles)
    # Non-monotonic scalar alias and independent thermal-frequency signatures.
    xcurve = np.linspace(.002, .95, 300)
    signatures = np.array([forward([x, 300, 100000, 0, 0, 0], frequencies)[:, 24] for x in xcurve])
    target = .23
    target_amp = abs(forward([target, 300, 100000, 0, 0, 0], frequencies[:1])[0, 24])
    right = brentq(lambda x: abs(forward([x, 300, 100000, 0, 0, 0], frequencies[:1])[0, 24]) - target_amp, xgrid[turning] + 1e-3, .98)
    np.savez_compressed(R / 'concentration_response.npz', x=xcurve, complex_phase=signatures, frequencies_Hz=frequencies, alias_low=target, alias_high=right)
    frequency = np.geomspace(100, 500000, 240)
    dynamics = np.array([forward([x, 300, 100000, 0, 0, 0], frequency)[:, 24] for x in [.02, .20, .50, .85]])
    np.savez_compressed(R / 'frequency_response.npz', frequencies_Hz=frequency, x=np.array([.02, .20, .50, .85]), complex_phase=dynamics)
    summary = {'status': 'executed synthetic computational benchmark', 'config': asdict(C), 'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'matplotlib', 'python-docx']}, 'python': sys.version, 'platform': platform.platform(), 'frequencies_Hz': frequencies.tolist(), 'calibration_scales': scales, 'discrepancy_thresholds': thresholds, 'main': overall, 'strata': stratified, 'bootstrap': boot, 'challenges': {scenario: metrics([row for row in challenge if row['scenario'] == scenario]) for scenario in challenge_arrays}, 'verification': verify, 'alias_low': target, 'alias_high': right, 'scalar_control_mae_pp': float(np.mean([abs(row['true_x'] - row['estimated_x']) for row in scalar]) * 100), 'calibration_n': 120, 'test_n': 320, 'profile_cases': chosen, 'joint_median_fit_seconds': float(np.median([row['time_s'] for row in test_records if row['method'] == 'Joint uncertainty'])), 'runtime_seconds': time.perf_counter() - start}
    (R / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
