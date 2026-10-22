# Physics-Informed-Hydrogen-Estimation-in-Hollow-Core-Fibre-Sensors
A physics-informed estimator quantifies hydrogen concentration in hollow-core fibre photothermal sensors under uncertain temperature, pressure, optical gain and molecular relaxation. Joint spectral and frequency-response inference reduces ambiguity at high H₂ concentrations while exposing sensitivity to model mismatch.

The observations and results are from an actually executed, fixed-seed Python computational benchmark. They are synthetic, not laboratory measurements. No COMSOL run, measured sensor, certified calibration gas or retrieved HITRAN line record is claimed.

## Study scope

Binary H₂/N₂; hydrogen mole fraction 0.002–0.95 (0.20%–95.00%); 270–330 K; 80–160 kPa. Pump/probe wavelengths are 2121.8/1550 nm. Truth uses 240 annular radial finite volumes; inversion uses 40 Bessel modes. Both share the continuum heat, spectroscopy and gas-property assumptions. The isothermal wall, Gaussian modes, line/broadening constants, linear mixture conductivity, refractivity and effective relaxation laws need experimental calibration before use in a real sensor.

Pure-gas 300 K conductivity endpoints use the NIST-hosted reference table identified in literature_matrix.json. Molecular line-strength, broadening and relaxation parameters are clearly identified surrogates. The model uses standard spectroscopic units and temperature laws, not downloaded spectroscopic line data.

The principal execution took 405.23 seconds in the recorded runtime; performance will vary. Python 3.12.14 was used. 

The root seed is 20261011; independent streams use +1 for calibration, +2 for testing, +100 onward for challenges, +600 for bootstrap

## Outputs and units

- physics.py: continuum closures, finite-volume truth, Bessel inverse and constrained estimator.
- run_study.py: frequency design; 120 calibration cases, 320 test cases, five estimators, four 64-case challenges, 2000 bootstrap replicates, and eight prespecified profiles.
- make_figures.py: original PNG/SVG pictorial schematics and scientific charts. All plotted data derive from retained arrays and tables.
- build_manuscript.py and word_math.py: native editable Word equations and manuscript authoring.
- verify_outputs.py: independent per-case metric recomputation, alias, profile, figure and DOCX integrity checks.
- results/*_data.npz: synthetic truth theta = (x,T,p,log_gain,log_relaxation,shift); auxiliary = (T,p,log_gain,log_relaxation); complex_phase dimensions (case,frequency,detuning); single_equal_time dimensions (case,1,detuning).
- Concentration x is a unitless mole fraction. Concentration displayed in percent is 100x. Absolute errors and interval widths are percentage points, not relative percentages.
- T is kelvin, p is pascals in CSV/NPZ (kilopascals in manuscript), frequency is hertz, phase is radians, line detuning is cm⁻¹, and thermal-field radius is metres.
- results/*_estimates.csv: per-case point estimates, SE, objectives, interval endpoints, residual, branch gap, coverage and reportability.
- results/challenge_data.npz: source_test_ids plus phase_0..3 and auxiliary_0..3; scenario mapping is challenge_scenarios.json.
- results/concentration_profiles.csv: fixed-x nuisance fits for cases 5,35,85,115,165,195,245,275. Each scan uses one nuisance initialization and does not certify a global solution.
- results/bootstrap_summary.json and bootstrap_replicates.csv: paired, stratified case resampling. Within each replicate, the same physical-case indices are used across methods.
- literature_matrix.json: verified bibliographic sources, read/access status, supported claims and primary URLs.
- manuscript_manifest.json: equation, figure and table inventory.
- results/integrity_audit.json: independent consistency results.

## Important interpretation

Joint MAE is 3.28 pp; plug-in MAE is 4.27 pp; equal-time single-frequency MAE is 6.15 pp on the same 320 physical cases. Joint target-95% interval coverage is 92.50%. The in-phase-only comparison gives no statistically resolved point-error disadvantage. The thermal/spectral challenge has 18.75% coverage and accepts 32 estimates with error above 5 pp. The review gate can fail under compensated model mismatch. These retained failures must not be omitted when discussing the results.

No ppm detection limit, temporal Allan noise, leak alarm performance or experimental accuracy is established. Equal-time comparison assumes ideal noise scaling with inverse square root of spectral dwell; electronics, stabilization and gas filling are not simulated.

