"""H2/N2 harmonic photothermal model and constrained uncertainty-aware inversion.

Line strength, broadening, refractivity and relaxation values are explicit
benchmark surrogates; this code does not download or claim HITRAN line data.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from dataclasses import dataclass, asdict
import numpy as np
from scipy.special import j0, j1, jn_zeros, voigt_profile
from scipy.linalg import solve_banded
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Configuration:
    seed: int = 20261011
    T0: float = 300.0
    p0: float = 100000.0
    radius: float = 15e-6
    pump_radius_fraction: float = .65
    probe_radius_fraction: float = .60
    length: float = 7.0
    pump_power: float = .20
    pump_wavelength: float = 2121.8e-9
    probe_wavelength: float = 1550e-9
    line_strength: float = 1e-26
    lower_energy: float = 118.5
    rotation_B: float = 59.3
    gamma_N2: float = .040
    gamma_H2: float = .030
    gamma_temperature_power: float = .70
    k_N2: float = .0259
    k_H2: float = .1866
    k_temperature_power: float = .72
    refractivity_N2: float = 2.90e-4
    refractivity_H2: float = 1.32e-4
    relaxation_tau: float = 2e-6
    self_relaxation_ratio: float = .15
    noise_rad: float = 15e-6
    rho_noise: float = .25
    sigma_T: float = 2.0
    sigma_p: float = 5000.0
    sigma_log_gain: float = .03
    sigma_log_relaxation: float = .10
    truth_cells: int = 240
    inverse_modes: int = 40


C = Configuration()
KB = 1.380649e-23
RGAS = 8.314462618
CLIGHT = 299792458.0
MH2 = 2.01588e-3 / 6.02214076e23
DETUNING = np.linspace(-.24, .24, 49)
CANDIDATES = np.array([.5, 2, 8, 32, 128]) * 1000


def partition(T):
    J = np.arange(14)
    nuclear = np.where(J % 2 == 0, 1, 3)
    return np.sum((2 * J + 1) * nuclear * np.exp(-1.4387769 * C.rotation_B * J * (J + 1) / T))


def spectroscopic(x, T, p, shift=0., broadening_factor=1.):
    nu0 = 1 / (C.pump_wavelength * 100)
    ratio = partition(C.T0) / partition(T) * np.exp(-1.4387769 * C.lower_energy * (1 / T - 1 / C.T0))
    ratio *= (1 - np.exp(-1.4387769 * nu0 / T)) / (1 - np.exp(-1.4387769 * nu0 / C.T0))
    sigma = nu0 / CLIGHT * np.sqrt(KB * T / MH2)
    gamma = broadening_factor * p / C.p0 * ((1 - x) * C.gamma_N2 + x * C.gamma_H2) * (C.T0 / T)**C.gamma_temperature_power
    phi = voigt_profile(DETUNING - shift, sigma, gamma)
    # Number density in cm^-3 and absorption in cm^-1, then convert to m^-1.
    absorption = 100 * (p / (KB * T) * 1e-6) * x * C.line_strength * ratio * phi
    return absorption


def gas_properties(x, T, p, k_factor=1.):
    k = ((1 - x) * C.k_N2 + x * C.k_H2) * (T / C.T0)**C.k_temperature_power * k_factor
    volumetric_cp = 3.5 * p / T
    refractivity = ((1 - x) * C.refractivity_N2 + x * C.refractivity_H2) * p / C.p0 * C.T0 / T
    tau = C.relaxation_tau * C.p0 / p * np.sqrt(C.T0 / T) / ((1 - x) + C.self_relaxation_ratio * x)
    return k, volumetric_cp, refractivity, tau


def gaussian_area(r, width, radius):
    return 2 * np.exp(-2 * r**2 / width**2) / (np.pi * width**2 * (1 - np.exp(-2 * radius**2 / width**2)))


class ModalThermal:
    def __init__(self, modes=C.inverse_modes, radius=C.radius, pump_fraction=C.pump_radius_fraction, probe_fraction=C.probe_radius_fraction):
        self.radius = radius
        self.beta = jn_zeros(0, modes)
        q, weights = np.polynomial.legendre.leggauss(320)
        r = (q + 1) * radius / 2
        dr = weights * radius / 2
        phi = j0(self.beta[:, None] * r[None] / radius)
        pump = gaussian_area(r, pump_fraction * radius, radius)
        probe = gaussian_area(r, probe_fraction * radius, radius)
        source = np.sum(phi * (r * pump * dr)[None], axis=1) * 2 / (radius**2 * j1(self.beta)**2)
        observed = np.sum(phi * (2 * np.pi * r * probe * dr)[None], axis=1)
        self.coeff = source * observed

    def transfer(self, k, cp, frequencies):
        denominator = k * (self.beta[:, None] / self.radius)**2 + 1j * 2 * np.pi * np.asarray(frequencies)[None] * cp
        return np.sum(self.coeff[:, None] / denominator, axis=0)


MODAL = ModalThermal()


def fv_thermal(k, cp, frequencies, cells=C.truth_cells, radius=C.radius, pump_fraction=C.pump_radius_fraction, probe_fraction=C.probe_radius_fraction, robin_length=0., return_field=False):
    dr = radius / cells
    r = (np.arange(cells) + .5) * dr
    faces = np.arange(cells + 1) * dr
    volume = np.pi * (faces[1:]**2 - faces[:-1]**2)
    face_k = k * 2 * np.pi * faces / dr
    lower = -face_k[1:-1]
    upper = -face_k[1:-1]
    diag = np.zeros(cells)
    diag[:-1] += face_k[1:-1]
    diag[1:] += face_k[1:-1]
    diag[-1] += k * 2 * np.pi * radius / (dr / 2 + robin_length)
    pump = gaussian_area(r, pump_fraction * radius, radius)
    probe = gaussian_area(r, probe_fraction * radius, radius)
    result, fields = [], []
    for f in np.asarray(frequencies):
        band = np.zeros((3, cells), dtype=complex)
        band[0, 1:] = upper
        band[2, :-1] = lower
        band[1] = diag + 1j * 2 * np.pi * f * cp * volume
        u = solve_banded((1, 1), band, pump * volume)
        result.append(np.sum(u * probe * volume))
        fields.append(u)
    if return_field:
        return np.array(result), r, np.array(fields)
    return np.array(result)


def forward(theta, frequencies, truth=False, extra=None):
    x, T, p, log_gain, log_relaxation, shift = np.asarray(theta)
    extra = extra or {}
    k, cp, refractivity, tau = gas_properties(x, T, p, extra.get('k_factor', 1.))
    alpha = spectroscopic(x, T, p, shift, extra.get('broadening_factor', 1.))
    # Average pump power along the cell; a stable weak-absorption limit.
    argument = alpha * C.length
    pump_average = C.pump_power * np.where(argument > 1e-14, -np.expm1(-argument) / np.maximum(argument, 1e-30), 1.)
    if truth:
        heat = fv_thermal(k, cp, frequencies, cells=extra.get('cells', C.truth_cells), pump_fraction=extra.get('pump_fraction', C.pump_radius_fraction), probe_fraction=extra.get('probe_fraction', C.probe_radius_fraction), robin_length=extra.get('robin_length', 0.))
    else:
        heat = MODAL.transfer(k, cp, frequencies)
    relaxation = 1 / (1 + 1j * 2 * np.pi * np.asarray(frequencies) * tau * np.exp(log_relaxation))
    if extra.get('extra_pole', 0):
        relaxation /= 1 + 1j * 2 * np.pi * np.asarray(frequencies) * extra['extra_pole']
    # Sign corresponds to a negative isobaric thermo-optic coefficient.
    phase = -2 * np.pi * C.length / C.probe_wavelength * refractivity / T * np.exp(log_gain) * (heat * relaxation)[:, None] * (alpha * pump_average)[None]
    return phase * np.exp(1j * extra.get('phase_error', 0.))


def whiten(z, sigma):
    output = np.array(z, dtype=complex, copy=True)
    output[..., 0] = z[..., 0] / sigma
    output[..., 1:] = (z[..., 1:] - C.rho_noise * z[..., :-1]) / (sigma * np.sqrt(1 - C.rho_noise**2))
    return output


def observation(theta, frequencies, rng, sigma=C.noise_rad, extra=None):
    clean = forward(theta, frequencies, truth=True, extra=extra)
    noise = rng.normal(0, sigma, clean.shape) + 1j * rng.normal(0, sigma, clean.shape)
    for j in range(1, len(DETUNING)):
        noise[:, j] = C.rho_noise * noise[:, j - 1] + np.sqrt(1 - C.rho_noise**2) * noise[:, j]
    intercept = rng.normal(0, 20e-6, len(frequencies)) + 1j * rng.normal(0, 20e-6, len(frequencies))
    slope = rng.normal(0, 40e-6, len(frequencies)) + 1j * rng.normal(0, 40e-6, len(frequencies))
    return clean + noise + intercept[:, None] + slope[:, None] * DETUNING[None]


class Inverse:
    """Fit x, T, p, log gain, log relaxation and line shift in scaled coordinates."""
    SCALE = np.array([1., 300., 100000., 1., 1., .04])
    LOW = np.array([1e-6, 250., 50000., -1., -1.5, -.08]) / SCALE
    HIGH = np.array([.98, 355., 210000., 1., 1.5, .08]) / SCALE

    def __init__(self, frequencies, mode='joint', sigma=C.noise_rad, sigma_T=C.sigma_T, sigma_p=C.sigma_p):
        self.frequencies = np.asarray(frequencies)
        self.mode = mode
        self.sigma = sigma
        self.sigma_T, self.sigma_p = sigma_T, sigma_p
        A = np.stack([np.ones(49), DETUNING])
        self.Q = np.linalg.qr(whiten(A, sigma).real.T)[0]
        self.variable = np.arange(6) if mode in ['joint', 'inphase'] else np.array([0, 3, 4, 5])

    def project(self, data):
        white = whiten(data, self.sigma)
        return white - (white @ self.Q) @ self.Q.T

    def fit(self, data, auxiliary, fixed_x=None, starts=None):
        auxiliary = np.asarray(auxiliary)
        base = np.array([.3, auxiliary[0], auxiliary[1], auxiliary[2], auxiliary[3], 0.]) / self.SCALE
        if self.mode == 'nominal':
            base[1:3] = [1., 1.]
        base = np.clip(base, self.LOW + 1e-8, self.HIGH - 1e-8)
        variable = self.variable[self.variable != 0] if fixed_x is not None else self.variable
        if fixed_x is not None:
            base[0] = fixed_x
        y = self.project(data)
        def residual(values):
            scaled = base.copy()
            scaled[variable] = values
            theta = scaled * self.SCALE
            signal = self.project(forward(theta, self.frequencies))
            error = y - signal
            measurement = error.real.ravel() if self.mode == 'inphase' else np.r_[error.real.ravel(), error.imag.ravel()]
            prior = [(theta[3] - auxiliary[2]) / C.sigma_log_gain, (theta[4] - auxiliary[3]) / C.sigma_log_relaxation]
            if self.mode in ['joint', 'inphase']:
                prior = [(theta[1] - auxiliary[0]) / self.sigma_T, (theta[2] - auxiliary[1]) / self.sigma_p] + prior
            return np.r_[measurement, prior]
        candidates = []
        starts = starts if starts is not None else ([.02, .22, .5, .78, .93] if fixed_x is None else [None])
        for start in starts:
            initial = base.copy()
            if start is not None:
                initial[0] = start
            fit = least_squares(residual, initial[variable], bounds=(self.LOW[variable], self.HIGH[variable]), x_scale='jac', max_nfev=120, ftol=1e-8, xtol=1e-8, gtol=1e-7)
            theta = base.copy()
            theta[variable] = fit.x
            cost = float(np.sum(fit.fun**2))
            candidates.append((cost, theta * self.SCALE, fit))
        candidates.sort(key=lambda item: item[0])
        cost, theta, fit = candidates[0]
        # Measurement projection removes two real modes in each used quadrature.
        q = 1 if self.mode == 'inphase' else 2
        n_measure = q * len(self.frequencies) * (len(DETUNING) - 2)
        n_prior = 4 if self.mode in ['joint', 'inphase'] else 2
        data_rss = float(np.sum(fit.fun[:-n_prior]**2))
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac, rcond=1e-12)
        ix = np.flatnonzero(variable == 0)
        se = float(np.sqrt(max(0., covariance[ix[0], ix[0]]))) if len(ix) else None
        alternatives = [item for item in candidates[1:] if abs(item[1][0] - theta[0]) > .05]
        branch_gap = min([item[0] - cost for item in alternatives], default=1e9)
        return {'theta': theta, 'se_x': se, 'cost': cost, 'red_residual': data_rss / max(1, n_measure - len(variable)), 'branch_gap': branch_gap, 'success': bool(fit.success), 'nfev': fit.nfev}


def auxiliary_measurement(theta, rng, sigma_T=C.sigma_T, sigma_p=C.sigma_p):
    return np.array([theta[1] + rng.normal(0, sigma_T), theta[2] + rng.normal(0, sigma_p), theta[3] + rng.normal(0, C.sigma_log_gain), theta[4] + rng.normal(0, C.sigma_log_relaxation)])
