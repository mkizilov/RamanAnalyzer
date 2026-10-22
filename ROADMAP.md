# RamanAnalyzer development roadmap

> Planning document — October 2026. Unchecked items are proposed work, not shipped
> features. Milestone 1 is the next priority; later milestones are a selectable
> backlog, not a release-date or total-hours commitment. Estimates will be refined
> after the first tests and implementation spikes.

## Direction

Make RamanAnalyzer a reliable, approachable **MIT-licensed Python toolkit for
hobbyists experimenting with their spectroscopy platforms**, and for people
learning Raman analysis from **free, openly licensed, non-personal spectra and
clearly labeled synthetic examples**.

Keep the workflow inspectable: import a spectrum, understand its axis, inspect
artifacts, choose preprocessing deliberately, fit a model, and examine the
residuals. Add convenient interfaces around that workflow rather than hide its
assumptions behind an automatic material-identification claim.

A major planned addition is a **reproducible synthetic Raman spectrum generator**:
start with simple analytical signals, then add material-inspired band patterns,
instrument effects, backgrounds, and detector noise. It will support numerical
software tests, teaching, and analysis-method development. It is not a project
for building, training, or evaluating machine-learning models.

The software remains MIT-licensed. Reference spectra and other third-party
material keep their own terms. Public access does not imply redistribution
permission, and new synthetic examples do not establish the provenance of older
experimental inputs.

## Navigation

- [Current baseline](#current-baseline)
- [Milestones and effort estimates](#milestones-and-effort-estimates)
- [Foundation and numerical correctness](#foundation-and-numerical-correctness)
- [Synthetic spectrum generator](#synthetic-spectrum-generator)
- [Material-inspired preset collection](#material-inspired-preset-collection)
- [Reproducible workflows and interfaces](#reproducible-workflows-and-interfaces)
- [Scientific validation and optional extensions](#scientific-validation-and-optional-extensions)
- [Data provenance and licensing](#data-provenance-and-licensing)
- [Release acceptance criteria](#release-acceptance-criteria)

## Current baseline

The repository currently contains a single main Python module, a notebook,
example spectra, and figures. Implemented capabilities include CSV import,
wavelength-to-Raman-shift conversion, averaging, cropping, despiking, ALS and an
experimental IarPLS baseline routine, iterative Voigt fitting, and plots.

The [README](README.md) includes a small inline synthetic example. That example
is **not yet a reusable generator or a validated material library**.

Concrete starting points visible in the current code and earlier smoke tests:

- The IarPLS stopping condition compares `w` with itself. Fixing that condition
  must be followed by a review of the algorithm and numerical edge cases.
- Despiking assumes a default index and has integer-intensity compatibility
  problems with pandas 3.
- The parser labels the axis `Wavenumber` even when the input contains wavelengths.
- Fitting helpers are duplicated; all-components-pruned and invalid-bound paths
  need explicit regression tests.
- Plot helpers generally display figures immediately rather than returning a
  reusable plotting interface.
- The repository has no maintained automated test suite or standard package
  metadata yet. The workflow diagram also advertises smoothing that the module
  does not provide.

Documentation work has exposed these limitations; it has not fixed them.

## Milestones and effort estimates

These are preliminary hands-on development estimates, including tests and short
usage documentation. They are not promised completion times or billing targets.
External permission requests and optional research are not included. Acceptance
criteria, not elapsed time, determine whether a milestone is complete.

| Milestone | Deliverable | Initial estimate | Dependency |
| --- | --- | --- | --- |
| **M1 — Reliable core** | Regression suite; baseline, despiking, input/unit, and fitting edge-case fixes; CI | **26–40 hours** | Current code |
| **M2 — Synthetic generator foundation** | Seeded analytical generator, simple instrument/noise models, ground-truth export, three teaching scenarios | **24–36 hours** | M1 testing foundation; some work can proceed alongside M1 |
| **M3 — Material-inspired examples** | First 4–6 reviewed presets, provenance records, scenario notebooks, documented approximation limits | **20–32 hours** | M2 and acceptable source/permission evidence |
| **M4 — Reproducible analysis and batch use** | Structured results, saved configuration, exports, CLI, error-tolerant batch processing | **24–40 hours** | Stable core result conventions |
| **M5 — Packaging and approachable interfaces** | Installable package, backward-compatible imports, improved plots/tutorials, optional interactive notebook | **20–32 hours** | M1 and M4 interfaces |
| **M6 — Advanced numerical work** | Uncertainty, calibration, repeat-series, and spatial-map extensions | Scope separately | Evidence from earlier milestones |

M1 and M2 together are an initial **50–76-hour planning envelope**, not a
commitment to complete every later item. M3–M6 should be selected and re-estimated
based on usefulness, scientific validity, and available maintenance time.

### M1 work breakdown

| Work package | Estimate | Completion evidence |
| --- | --- | --- |
| Tests and synthetic fixtures | 4–6 h | Reproducible reference cases, including failures |
| Baseline correctness and stability | 5–8 h | Convergence regression and finite-result checks |
| Despiking correctness and compatibility | 4–6 h | Arbitrary-index, integer, flat-signal, and boundary tests |
| Import validation and axis units | 4–6 h | Explicit units and actionable malformed-input errors |
| Fitting robustness and duplicate cleanup | 5–8 h | No-peak/pruned-model/bounds tests; behavior-preserving refactor |
| CI, executable examples, and guidance | 4–6 h | Tests run in CI; quickstart stays valid |

### Suggested implementation order

1. Capture current behavior in tests before changing numerical routines.
2. Fix correctness bugs before adding automatic conveniences.
3. Build a minimal generator with arbitrary analytical peaks and known truth.
4. Validate each instrument/noise effect independently before combining them.
5. Add named material presets only after their sources and limitations are recorded.
6. Make the stable workflow exportable, batchable, and easy to install.
7. Consider a GUI and advanced scientific extensions only after the core is reliable.

## Foundation and numerical correctness

### F1. Automated tests and reproducible fixtures

- [ ] Add unit tests for parsing, conversion, cropping, averaging, baseline correction, despiking, and fitting.
- [ ] Turn the README synthetic workflow into an end-to-end regression test with justified numerical tolerances.
- [ ] Create small deterministic fixtures for isolated, overlapping, broad, weak, and out-of-window peaks.
- [ ] Cover empty and short arrays, constant signals, NaNs, infinities, negative corrected intensities, and missing columns.
- [ ] Test descending, duplicated, irregular, and non-overlapping axes explicitly.
- [ ] Verify that transformations preserve raw inputs unless mutation is explicitly documented.
- [ ] Exercise plotting with a non-interactive backend and close figures after tests.
- [ ] Add CI for a documented Python/dependency matrix; expand support only after it passes.

**Acceptance:** the main workflow and known failure cases can be reproduced without
external datasets, network access, or interactive plot windows.

### F2. Baseline correction

- [ ] Replace the IarPLS self-comparison with a meaningful convergence criterion and a failing-then-passing regression test.
- [ ] Review the full weight update against the intended method, documenting its reference and any deliberate variation.
- [ ] Handle zero residual variance, zero weight norms, exponential overflow, and non-finite intermediate values.
- [ ] Validate smoothing strength, asymmetry, iteration count, tolerance, and minimum input length.
- [ ] Review ALS behavior for exact fits and equal-to-baseline samples instead of leaving zero-weight cases implicit.
- [ ] Optionally return baseline values, corrected signal, iteration count, convergence status, and warnings.
- [ ] Compare results on sloped, curved, fluorescent, and broad-band synthetic backgrounds with known truth.
- [ ] Add a parameter-sensitivity notebook showing both useful correction and over-subtraction failure cases.

**Acceptance:** valid inputs produce finite results or an explicit failure status;
constant signals and broad peaks have documented behavior; convergence tests do
not merely assert that the routine returns an array.

### F3. Cosmic-ray despiking

- [ ] Use positional numeric arrays internally so sliced or non-default DataFrame indices work correctly.
- [ ] Process integer counts safely in floating point and test supported pandas versions.
- [ ] Define behavior when the median absolute deviation is zero or nearly zero.
- [ ] Audit how a flagged first difference maps to the affected sample or samples.
- [ ] Test leading/trailing spikes, adjacent spikes, wider artifacts, and windows with no clean neighbors.
- [ ] Return an optional artifact mask, repaired-point count, and unresolved-point warnings.
- [ ] Add a preview mode that identifies candidate artifacts without changing intensities.
- [ ] Quantify false corrections of narrow real peaks and document conservative defaults.

**Acceptance:** correction is index-independent, does not crash on flat signals,
and reports what it changed. Tests include legitimate narrow peaks that should
remain intact.

### F4. Import, units, and validation

- [ ] Represent wavelength in nm and Raman shift in cm^-1 explicitly; keep a compatibility path for existing DataFrame callers.
- [ ] Support both the existing multi-trace format and ordinary two-column CSV input.
- [ ] Make delimiter, skipped metadata rows, header handling, and selected columns configurable.
- [ ] Report malformed rows and invalid numbers with enough file/column context to fix the input.
- [ ] Define sorting and duplicate-coordinate policies rather than silently accepting ambiguous interpolation inputs.
- [ ] Validate excitation wavelength and distinguish Stokes from anti-Stokes coordinates.
- [ ] Prevent accidental double conversion by carrying axis metadata through the workflow.
- [ ] Document the difference between counts per detector bin and a spectral density when converting or resampling axes.

**Acceptance:** a wavelength trace cannot silently masquerade as a Raman-shift
trace. Existing supported CSV examples continue to work or have an explicit
migration path.

### F5. Preprocessing and repeated measurements

- [ ] Define named preprocessing steps and preserve their order and settings.
- [ ] Add optional Savitzky–Golay smoothing with validated window length, order, and sampling assumptions.
- [ ] Add optional max, area, and selected-band normalization with zero-denominator checks.
- [ ] Document how normalization and smoothing affect intensity comparisons and noise statistics.
- [ ] Return the number of contributing repeats and optional spread/error summaries when averaging.
- [ ] Reject absent overlap; expose the interpolation grid and avoid unintended extrapolation.
- [ ] Allow inspection and export of intermediate spectra without rerunning the pipeline.
- [ ] Add explicit policies for missing samples, saturated samples, and invalid detector bins.

**Acceptance:** processing is reproducible from a saved ordered configuration, and
users can trace a fitted signal back to its unmodified input.

### F6. Voigt fitting and result status

- [ ] Remove duplicate helper definitions without changing the public API unexpectedly.
- [ ] Share initial-peak and residual-peak construction/pruning logic instead of maintaining parallel copies.
- [ ] Handle no detected peaks and all components rejected without fitting an empty model.
- [ ] Validate initial guesses, amplitude/width bounds, center windows, and downsampling settings.
- [ ] Audit whether removed parameters can be reintroduced during later parameter updates.
- [ ] Add optional user-supplied centers, fit windows, fixed parameters, and maximum component counts.
- [ ] Require a documented improvement rule before adding residual peaks; prevent near-duplicate components.
- [ ] Report optimizer status, stop reason, bound hits, residual diagnostics, and parameter-correlation warnings.

**Acceptance:** a failed or underdetermined fit is distinguishable from a successful
fit, and noise-only or fully pruned inputs do not cause an unhandled exception.

## Synthetic spectrum generator

### Purpose and realism levels

The generator should make it possible to ask: **“I know what signal I generated;
how much of it did the analysis recover, and what distorted it?”**

Use three explicit levels rather than calling every simulated curve “realistic”:

1. **Analytical fixtures:** arbitrary peak positions and exact mathematical
   parameters for tests. These do not represent a named material.
2. **Material-inspired recipes:** reviewed band patterns and approximate relative
   strengths, with documented sources and deliberate simplifications.
3. **Reference-informed simulations:** optional fits or transformations of a
   specific reusable reference spectrum, retaining its identity, conditions,
   license, and provenance. These remain simulations, not new measurements.

Initially, implement level 1. Level 2 follows only with reviewed source records.
Level 3 is optional and depends on permission and adequate measurement metadata.

A preset must not imply universal peak intensities, certified calibration values,
reliable chemical identification, or first-principles prediction of arbitrary
materials. Excitation, orientation, polarization, temperature, phase, focus, and
instrument response can all change a real spectrum.

### G1. Generator API, configuration, and ground truth

- [ ] Define a versioned configuration schema with explicit axis, units, range, bin edges, and intensity conventions.
- [ ] Support a local random seed and independently derived random streams for peaks, backgrounds, detector noise, and artifacts.
- [ ] Validate configuration before allocating arrays; cap unreasonable sample and batch sizes.
- [ ] Return the clean intrinsic signal, instrument-broadened signal, expected counts, noisy observation, component curves, and artifact masks separately.
- [ ] Record ground-truth centers, intrinsic widths, areas, simulated drift, background, and detector parameters.
- [ ] Include generator version, recipe version, random seed, numerical environment, and source records in exported metadata.
- [ ] Allow a user-defined analytical peak list without requiring a named material or an external download.
- [ ] Keep generation usable independently of preprocessing and fitting so errors do not become circular tests of one shared implementation.

**Acceptance:** the same configuration and seed reproduce the same realization
within the documented environment. Random sequences need not remain bitwise
identical across arbitrary dependency versions; record those versions explicitly.

### G2. Peak profiles and mixtures

- [ ] Support Gaussian, Lorentzian, and Voigt profiles with unambiguous area-versus-height normalization.
- [ ] Define standard deviation, Gaussian FWHM, Lorentzian HWHM, and observed Voigt width consistently.
- [ ] Evaluate on an adequately sampled internal grid and integrate into output bins where required.
- [ ] Preserve full-profile versus finite-window area distinctions for truncated tails and boundary peaks.
- [ ] Generate isolated peaks, shoulders, overlapping multiplets, broad bands, and weak peaks near strong ones.
- [ ] Permit bounded per-realization variation in center, width, and strength, recording the realized values.
- [ ] Add optional linear mixtures of recipe signals with explicit nonnegative signal weights.
- [ ] Label mixture weights as simulated signal contributions, not mole/mass fractions or calibrated concentrations.

**Acceptance:** profile integrals and width conventions agree with independent
analytical/reference checks; truncated spectra do not silently claim to contain
the full peak area.

### G3. Optical response and sampling

- [ ] Separate intrinsic material line shape from instrumental line-spread broadening to avoid counting resolution twice.
- [ ] Start with a normalized Gaussian instrument line-spread function; make more complex kernels an optional later extension.
- [ ] Convolve on a padded grid without circular wraparound, then crop and bin deliberately.
- [ ] Add optional detector spectral sensitivity and optical/filter transmission curves with documented units.
- [ ] Support an explicit simplified low-shift cutoff rather than pretending it models a particular commercial filter.
- [ ] Simulate axis offset, scale error, and modest nonlinear distortion while preserving both true and reported coordinates.
- [ ] Support wavelength-domain detector bins and Raman-shift output, respecting integration and the spectral-density Jacobian where applicable.
- [ ] Separate experiment parameters from phenomenological controls: do not imply that changing excitation wavelength predicts resonance, fluorescence, or absolute scattering strength automatically.

**Acceptance:** zero broadening approaches the clean signal, normalized broadening
conserves area away from cropped boundaries, and axis conversion does not
accidentally rescale integrated photon counts.

### G4. Backgrounds, noise, and detector artifacts

- [ ] Add constant, sloped, low-order polynomial, and broad fluorescence-like backgrounds with physically valid expected counts.
- [ ] Distinguish optical background from electronic offset; document where each enters the simulation.
- [ ] Model shot noise on expected photoelectrons and dark current before electronic read noise and output gain conversion.
- [ ] Make Gaussian additive noise a separate, explicitly simplified mode for normalized arbitrary-unit signals.
- [ ] Simulate exposure-time changes without suggesting that longer acquisition removes systematic artifacts.
- [ ] Add optional read noise, gain/offset, quantization, full-well saturation, and ADC clipping with explicit processing order and masks.
- [ ] Simulate sparse cosmic-ray-like detector events and persistent hot/dead pixels separately; record their affected samples.
- [ ] Add optional structured background/fringe stress cases, clearly labeled as phenomenological rather than universal instrument models.

**Acceptance:** repeated realizations have the expected statistical behavior in
simple cases; zero-noise mode is deterministic; shot noise is never sampled from
negative expectations; clipped measurements are not treated as uncensored data.

### G5. Sequences, repeatability, and exposure scenarios

- [ ] Generate repeated acquisitions with independent random noise and shared persistent detector defects.
- [ ] Add smooth or abrupt axis drift, intensity drift, and background changes across a sequence.
- [ ] Provide low-count, moderate-count, high-background, saturated, and drift-dominated scenarios.
- [ ] Support optional photobleaching-like intensity/background trends as documented phenomenological scenarios.
- [ ] Generate pairs of measurements that differ in only one controlled effect for teaching and regression tests.
- [ ] Save per-exposure truth rather than only the nominal starting parameters.
- [ ] Use deterministic per-spectrum seeds so batch ordering or parallel scheduling does not change generated examples.
- [ ] Leave synthetic spatial maps for a later milestone after the single-spectrum and repeated-acquisition APIs stabilize.

**Acceptance:** repeating or reordering a batch does not alter a spectrum's identity,
and averaging tests can distinguish random-noise reduction from persistent bias.

### G6. Export, visualization, and numerical validation

- [ ] Export ordinary CSV and the existing multi-trace parser layout for round-trip tests.
- [ ] Add JSON sidecars for configuration, component truth, artifact masks, and provenance.
- [ ] Separate observation files from truth files so analysis tests do not accidentally use the answers.
- [ ] Provide plots comparing intrinsic signal, instrument response, background, observation, and recovered fit.
- [ ] Add tests for profile area, bin integration, convolution boundaries, noise moments, and seed reproducibility.
- [ ] Evaluate recovery against both intrinsic and instrument-broadened parameters where they differ.
- [ ] Use held-out scenarios and alternative line shapes to avoid testing only signals identical to the fitter's assumptions.
- [ ] Keep a small offline fixture set in the repository; generate larger numerical stress suites on demand rather than committing bulky outputs.

**M2 minimum deliverable:** arbitrary analytical peaks, a seedable generator,
constant/curved backgrounds, fixed Gaussian resolution, basic shot/read-noise
modes, truth export, and three teaching scenarios. Full detector behavior,
sequences, wavelength-domain sampling, and every checkbox above are later scope,
not hidden requirements inside the 24–36-hour M2 estimate.

## Material-inspired preset collection

### Candidate materials and teaching purpose

These are **proposed preset subjects**, not a claim that verified spectra or
approved band tables are already bundled. Exact band positions, relative
strengths, conditions, and reuse terms must be reviewed before implementation.
No commercial spectral library or paywalled table will be copied without rights.

| Candidate | Educational purpose | Important limitation |
| --- | --- | --- |
| Silicon | Narrow-band recovery, instrumental broadening, axis-offset exercises | Position/shape depend on conditions; not a certified calibration reference |
| Polystyrene | Multi-band fitting and relative peak-position checks | Do not imply standard-grade material or universal relative intensities |
| PTFE / Teflon | Polymer band patterns and a synthetic counterpart to the existing example workflow | A new recipe does not establish the source or permission for `data/teflon.csv` |
| Polyethylene | Overlapping polymer bands and resolution sensitivity | Crystallinity, orientation, and grade affect spectra |
| Polypropylene | Similar-looking polymer patterns and ambiguous fits | A good fit is not a reliable automatic polymer identification |
| PMMA | Fluorescence-background and weak-band exercises | Additives and excitation conditions can change the background |
| PET | Denser multi-band examples and overlap | Orientation/crystallinity must not be hidden as random universal variation |
| Calcite | Mineral bands and relative-width exercises | Grain orientation and polarization influence relative strengths |
| Quartz | Multiple narrow mineral features | Phase, orientation, and acquisition conditions matter |
| Gypsum | Closely spaced mineral features and window selection | Hydration and phase cannot be inferred from a generic recipe alone |
| Graphitic/disordered carbon | Broad-band fitting and line-shape mismatch | D/G-like features require careful excitation/structure-dependent interpretation |
| An approved two-component mixture | Component overlap and recoverable signal weights | Signal weights are not chemical concentration without calibration |

### P1. Preset record and review process

- [ ] Give each preset a stable identifier, version, material description, and realism level.
- [ ] Record source URLs/DOIs, reference identifiers, acquisition conditions when known, and access dates.
- [ ] Record redistribution/derivative-use terms separately from the project's MIT software license.
- [ ] Separate sourced band positions from chosen illustrative widths, intensities, and background parameters.
- [ ] Document unsupported conditions rather than extrapolating to arbitrary excitation wavelengths or temperatures.
- [ ] Include a review checklist for units, line-shape conventions, duplicate bands, bounds, and expected plotted appearance.
- [ ] Label every plot/export as synthetic and name the recipe; retain attribution for any reference-informed material.
- [ ] Begin with 4–6 well-documented examples rather than rushing to a large, weakly justified library.

### P2. Teaching scenarios built from presets

- [ ] Compare a narrow-band material at several instrumental resolutions.
- [ ] Show how overlapping polymer bands become difficult to fit independently.
- [ ] Demonstrate weak peaks on a strong fluorescence-like background.
- [ ] Contrast cosmic-ray artifacts with legitimate sharp bands.
- [ ] Compare independent read/shot noise with persistent hot pixels across repeated measurements.
- [ ] Demonstrate how clipping can make peak-height and width estimates misleading.
- [ ] Show calibration drift shifting an otherwise unchanged material-inspired signal.
- [ ] Include a deliberately mismatched fitting model to demonstrate that low residual error is not a guarantee of correct physics.

### Reference discovery, not blanket permission

Potential starting points include the [RRUFF mineral database](https://www.rruff.net/)
and [OpenSpecy](https://www.openspecy.org/). These links are suggestions for source
review, **not** assertions that every hosted spectrum can be copied, relicensed,
or used without attribution. Inspect the particular record and applicable terms.
Publications and calibration standards can guide the scientific review without
copying restricted tables or claiming certified calibration performance.

**M3 acceptance:** each released preset has traceable parameters, a documented
permission basis where third-party content is used, a plotted review example,
and a plain-language limitations note. Unsupported presets stay proposed.

## Reproducible workflows and interfaces

### W1. Structured results and export

- [ ] Define a result object containing processed data, baseline, fit/components, peak table, warnings, and configuration.
- [ ] Export centers, areas, heights, widths, units, and available parameter uncertainties to a stable tabular format.
- [ ] Preserve the original fit result for advanced users rather than replacing it with a lossy summary.
- [ ] Save input checksums, relevant software versions, and preprocessing order.
- [ ] Save and reload validated analysis configurations with schema-version handling.
- [ ] Export CSV and JSON without requiring interactive plotting.
- [ ] Produce a compact optional HTML report containing plots, settings, warnings, and source information.
- [ ] Define explicit schema and migration rules before downstream users depend on serialized results.

### W2. Batch analysis and command-line use

- [ ] Process one file or a directory using the same configuration as the Python API.
- [ ] Allow explicit per-file overrides without silently changing batch-wide units.
- [ ] Continue past failed files and produce a machine-readable failure report.
- [ ] Generate a summary table with fit status and quality flags alongside successful peak estimates.
- [ ] Add progress reporting, cancellation, and deterministic output naming.
- [ ] Support resumability using input/configuration identity, not merely the existence of an output file.
- [ ] Add a CLI for analysis, generation, and report export once the corresponding Python interfaces are stable.
- [ ] Profile realistic workloads before adding optional parallel execution; test ordering, cancellation, and memory use.

### W3. Plotting and approachable interaction

- [ ] Return figure/axes objects and make display optional.
- [ ] Fix empty legends, apply correct axis labels, and carry units into exports.
- [ ] Provide before/after preprocessing plots with changed points and estimated baseline.
- [ ] Improve total-fit, component, and residual displays, including bound/failure warnings.
- [ ] Support configurable figure size, resolution, output format, and accessible color/line styles.
- [ ] Add an interactive notebook for previewing processing settings on synthetic examples.
- [ ] Create a tutorial from a plain two-column export to a peak table, explaining every processing decision.
- [ ] Explore a small optional local GUI only after the numerical API is stable; do not make it a mandatory dependency.

### W4. Packaging and maintenance

- [ ] Add `pyproject.toml`, dependency declarations, and local editable-install instructions.
- [ ] Split parsing, preprocessing, baseline, fitting, plotting, and generation into focused modules.
- [ ] Preserve the established `import RamanAnalyzer as ra` usage through a documented compatibility layer where feasible.
- [ ] Add type hints and consistent docstrings to public interfaces.
- [ ] Replace unconditional prints with configurable logging and structured warnings.
- [ ] Remove tracked generated artifacts through normal commits, not history rewriting.
- [ ] Document tested Python/dependency ranges and remove compatibility caps only after passing tests.
- [ ] Add a changelog, release checklist, and versioned examples while preserving the MIT notice and original history.

## Scientific validation and optional extensions

### V1. Measure analysis quality

- [ ] Measure recovery error for peak positions, areas, heights, and widths separately.
- [ ] Measure false peak detections on noise-only signals and missed peaks near strong neighbors.
- [ ] Quantify distortion introduced by baseline correction, despiking, smoothing, interpolation, and downsampling.
- [ ] Test model mismatch rather than fitting only exact Voigt signals with a Voigt model.
- [ ] Characterize behavior across overlap, resolution, sampling density, background, and count-level sweeps.
- [ ] Distinguish actual spectral resolution from interpolation or denser plotting grids.
- [ ] Keep deterministic fast tests separate from repeated statistical tests with documented tolerances.
- [ ] Validate selected synthetic findings against permission-cleared experimental examples before making real-world accuracy claims.

These are evaluations of numerical spectroscopy methods, not AI-model training
or AI-model benchmarks. Simulation performance alone does not establish material
identification, calibration certification, or clinical suitability.

### V2. Uncertainty and model comparison

- [ ] Expose parameter correlations, uncertainty availability, and bounds explicitly.
- [ ] Add optional measurement uncertainties/weights with a documented noise model.
- [ ] Explain how baseline subtraction and smoothing change covariance assumptions.
- [ ] Evaluate bootstrap or repeated-measurement uncertainty approaches appropriate to those assumptions.
- [ ] Compare candidate peak counts using residual structure and justified model-selection diagnostics.
- [ ] Define reduced-statistic labels and effective degrees of freedom carefully; do not equate optimizer success with a valid model.
- [ ] Check uncertainty coverage on independent synthetic realizations, including failures and boundary solutions.
- [ ] Show when overlapping components are not individually identifiable and recommend reporting a simpler result.

### V3. Optional later scientific features

- [ ] User-supplied wavelength calibration curves with residual checks and validity ranges.
- [ ] Propagation of wavelength and excitation-wavelength uncertainty into Raman shift.
- [ ] Explicit dark/background subtraction with scaling and uncertainty assumptions.
- [ ] User-provided instrument-response correction with masked or near-zero sensitivity handling.
- [ ] Peak tracking across repeat/time/temperature series with ambiguous matches reported rather than forced.
- [ ] Alternative Gaussian/Lorentzian/Voigt models with comparable parameter conventions.
- [ ] Spatial parameter maps built on a reliable batch result format and recorded scan coordinates.
- [ ] A small importer extension interface for documented additional formats, with test fixtures and source permissions.

Each extension requires a separate design and effort estimate. The project is not
committing to instrument-control software, a first-principles materials simulator,
an automated chemical-identification service, or a cloud data-collection system.

## Data provenance and licensing

### D1. Existing examples

- [ ] Inventory `data/teflon.csv`, notebook inputs/outputs, and example figures.
- [ ] Record where each experimental input came from and who may authorize its redistribution.
- [ ] Attach a public source/license or an accurate permission statement when supported by evidence.
- [ ] Distinguish measured examples from synthetic examples, including figures derived from either.
- [ ] Resolve unclear rights before using an example in new shared workflows; use independent analytical fixtures for tests in the meantime.
- [ ] Record factual historical context separately from current intended use; do not rewrite old commits to imply a different origin.
- [ ] Keep required third-party attribution without reintroducing a mandatory citation demand for using the MIT-licensed code.
- [ ] Keep personal, customer, regulated, confidential, and secret material out of new fixtures and documentation.

### D2. New generator outputs and references

- [ ] Give original analytical recipes an explicit output-reuse policy instead of assuming the code license answers every data question.
- [ ] Preserve source and derivative-use obligations for reference-informed outputs.
- [ ] Keep clearly synthetic fixtures small and self-contained; provide scripts for reproducing larger sets.
- [ ] Require a source/permission review for each external dataset, spectrum, digitized figure, or band table considered for inclusion.
- [ ] Do not label an example “open data” merely because a download URL is accessible.
- [ ] Track generated-data versions so changing a recipe does not silently change the meaning of old test results.

The roadmap is a development plan, not a certification of ownership, historical
funding, or existing dataset permissions. Those facts require their own accurate
records. Adding a synthetic replacement does not answer where an older example
came from.

## Release acceptance criteria

For each selected work package:

1. Define the behavior and at least one failure case before implementation.
2. Add a regression test for the original bug or a measurable test for the feature.
3. Keep units, input mutation, randomness, and return types explicit.
4. Demonstrate the behavior on a self-contained synthetic example.
5. Record numerical assumptions and known limitations, including misleading cases.
6. Check compatibility with existing scripts or provide migration guidance.
7. Update the relevant user documentation and mark only verified work complete.
8. Preserve MIT notices, dataset terms, and the repository's real commit history.

### First generator release checklist

- [ ] No external data download is required for analytical examples.
- [ ] Inputs and outputs distinguish intrinsic signal, instrument effects, and observed data.
- [ ] Seeds and version metadata support reproducibility.
- [ ] Area/width conventions and finite-window effects are tested.
- [ ] Noise statistics and convolution boundaries have independent checks.
- [ ] Synthetic observations and ground truth export separately and round-trip correctly.
- [ ] Every named material recipe is either reviewed and documented or explicitly unavailable/proposed.
- [ ] Documentation says what is simulated and what is not scientifically justified.

### Scope discipline

Keep the first release small enough to review. Prefer a few correct, documented
behaviors to a large list of unreliable options. Use later milestones as a place
to record useful ideas, then promote individual items only when there is a real
use case, a testable outcome, and time to maintain them.
