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

**Purpose**: starting from a low-threshold TPs radiological background sample, calculate the expected TP rages from radiological backgrounds as a function of `adc_peak` threshold.
Extract the per-plane TP thresholds corresponding to O(100) kHz TP rates per CRP plane, separating the contributions from physical backgrounds and electronic noise.

1. Calculate trigger rates per plane as a function of `adc_peak` thresholds
2. Calculate signal completeness vs `adc_peak` for a single-electron sample (note: background backtracking is not reliable)
3. Estimate signal/noise ratio for reference threshold  (Slightly above 100 kHz)


## `VD-1x8x14-TPCalibration`

Calibration of TP integrated charge using single electron and gamma samples.

**Purpose**: Calculation of calibration coefficients to Trigger Primitive's `adc_intergral` for different planes using single electron and single gamma datasets
(most relevant for Low Energy physics)


## `VD-1x8x14-TPFiltering`

TriggerPrimitive filtering is applied before the TriggerActvity making stage to remove TPs asso



# Obsolete / not maintained


## Outdated : `VD-1x8x14-NoiseRateEstimation`

Short studies on the predicted rates based on the raw ADC distributions.
The notebook shows how the SimChannel association to waveforms is not fully reliable in the radiological backgrounds sample
