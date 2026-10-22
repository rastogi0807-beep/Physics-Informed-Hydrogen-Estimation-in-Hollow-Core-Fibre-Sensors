# Fixed-seed computational study protocol

Question: Can spectral complex photothermal observations at three modulation frequencies resolve hydrogen concentration in a binary H2/N2 gas despite uncertain temperature, pressure, optical gain and effective collisional relaxation?

Status: Computational model study. No laboratory sensor, commercial solver run or certified gas calibration has been supplied. Physical constants and molecular line/relaxation surrogates must be distinguished from externally retrieved values. No trace-gas detection-limit claim is planned.

Mechanics: weak quadrupole absorption near 2121.8 nm; Gaussian pump and probe in a 15 micrometre radius hollow core; harmonic radial heat equation with an isothermal boundary; isobaric gas refractivity; one effective collisional relaxation pole. Truth uses a radial finite-volume discretization; inversion uses a Bessel eigenfunction expansion. Both share continuum physics.

Design: concentration mole fraction 0.002–0.95, four equal-count strata; temperature 270–330 K; pressure 80–160 kPa. Three frequencies selected from 0.5, 2, 8, 32 and 128 kHz by minimizing the worst marginal Fisher standard error over an independent 18-point design grid. The main test set is not used for this selection. 49 spectral samples at detunings -0.24 to 0.24 cm^-1.

Noise: independent Gaussian real/imaginary phase noise, 15 microradians per quadrature before AR(1) spatial-in-spectrum correlation rho=0.25; constant and linear complex background per frequency. Estimated temperature and pressure are separate noisy observations with 2 K and 5 kPa standard deviations. Log-gain and log-relaxation calibrations have uncertainties 0.03 and 0.10. Their actual gains/scales vary across cases. Calibration and test seeds differ.

Comparisons: nominal-environment fit; plug-in measured T/P fit; jointly uncertain T/P fit; in-phase-only fit; single-frequency joint fit at equal total spectral dwell, using noise divided by sqrt(3). A scalar lower-branch peak calibration is a deliberately limited ambiguity control, not the only baseline.

Counts: 120 calibration cases, 320 independent test cases, all four concentration strata balanced. Per-method interval scales and nominal discrepancy threshold use calibration only. Concentration accuracy uses absolute mole-fraction errors reported as percentage points. Interval coverage, median width, raw versus scaled coverage, branch ambiguity and accepted wrong estimates are reported separately. Paired case bootstrap uses 2000 replicates with stratification.

Checks: steady heat solution, radial mesh convergence, modal truncation convergence, finite-volume/eigenfunction agreement, dimensions and small temperature rise; saved-result consistency checks. Challenge tests include coherent nuisance uncertainty changes, biased auxiliary measurements, thermal/line/relaxation model mismatch and unknown interferometer phase. Failure evidence is retained rather than filtered out of the headline accuracy.

Uncertainty: local covariance includes auxiliary measurement residuals. Calibration-scaled intervals are conditional on the specified surrogate model and finite calibration set. Selected global concentration profile scans test whether local intervals miss alternative branches. Neither establishes universal frequentist coverage under misspecification.

Outputs: reproducible Python code, generated complex observations and auxiliary measurements, configurations, estimates, metrics, confidence profiles, sensitivity/design tables, original PNG/SVG figures and a Word manuscript. Author metadata and declarations require completion before submission. No prospective registration is claimed.
