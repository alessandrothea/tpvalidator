# Notebooks

## `VD-1x8x14-RadBkgMontecarlo`

**Purpose**: Perform a basic characterization of radiological background objects:
- List of generators
- Total MC object rates
- Activity (counts) by generator (sorted by activity)
  * for `gammas` and `electrons`
- MC particles spectra (top 10)
- MC origin (time and xyz coodinates in the detector)

Obsolete (to be moved out)
- TP origin by generator

## `VD-1x8x14-RadBkgTriggerPrimitiveRates`

Definition of baseline TP threshold (Simple Threshold algorithm) using radiological background rates calculation

**Purpose**: Calculate the expected TP rages from radiological backgrounds as a function of `adc_peak` threshold, starting from a low-threshold TPs radiological background sample.
Extract the per-plane TP thresholds corresponding to O(100) kHz TP rates per CRP plane, separating the contributions from physical backgrounds and electronic noise.

1. Calculate trigger rates per plane as a function of `adc_peak` thresholds
2. Calculate signal completeness vs `adc_peak` for a single-electron sample (note: background backtracking is not reliable)
3. Estimate signal/noise ratio for reference threshold  (Slightly above 100 kHz)


## `VD-1x8x14-TPCalibration`

Calibration of TP integrated charge using single electron and gamma samples.

**Purpose**: Calculation of calibration coefficients to Trigger Primitive's `adc_intergral` for different planes using single electron and single gamma datasets
(most relevant for Low Energy physics)


## `VD-1x8x14-TPFiltering`

TriggerPrimitive filtering is applied before the TriggerActvity making stage to remove TPs associated to very low energy deposits such as Ar39.
This is accomplished by applying tighter `adc_peak` cuts and an additional `samples_over_threshold`.

**Purpose**: Assess the impact of TP filtering cuts on signals and radiological samples

- Display the `adc_peak`/`samples_over_threshold` distributions
  - `adc_integral` weighting is used together with TP counts to highlight where the 
- Show the impact of incremental of the additional `samples_over_threshold` cut on distributions, for signals (e- and gamma primarily) and backgronds
- Show the impact on rates per background type and confirm it meets the expectations 


# Obsolete / not maintained


## Outdated : `VD-1x8x14-NoiseRateEstimation`

Short studies on the predicted rates based on the raw ADC distributions.
The notebook shows how the SimChannel association to waveforms is not fully reliable in the radiological backgrounds sample
