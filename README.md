# lazyfmri | utility package for fMRI analyses

![plot](https://github.com/gjheij/lazyfmri/blob/main/data/example.png)

This package contains utility functions used for routines such as GLMs, deconvolution, and plotting often used in the [linescanning](https://github.com/gjheij/linescanning)-repository. 
This is separate to make functions more easily available without installing the entire pipeline.

## Installation

```bash
pip install git+https://github.com/gjheij/lazyfmri
```

## Functionalities

The main functions of this package include plotting, estimation of the hemodynamic response function (HRF) through various methods, and basic preprocessing of 2D fMRI data.

I have been using these functions for several of my publications:

- ["A selection and targeting framework of cortical locations for line-scanning fMRI"](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.26459)
- ["Quantitative MRI at 7-Tesla reveals novel frontocortical myeloarchitecture anomalies in major depressive disorder](https://www.nature.com/articles/s41398-024-02976-y)

### I/O

The [Dataset](https://github.com/gjheij/lazyfmri/blob/main/lazyfmri/dataset.py#L2943)-class can parse several file types typically used in fMRI research:

- Functional files: `*.mat`, `*.nii.gz`, or `*.gii` and outputs a 2D dataframe (time, datapoints) indexed by  given (or extracted) subject/run/task IDs.
These IDs can either be given or extracted from the filename if the filename follows the [BIDS](https://bids.neuroimaging.io/index.html)-format.

![plot](https://github.com/gjheij/lazyfmri/blob/main/data/df_func.png)

- Experiment files: this package is designed to work with the `*.tsv`-output from the [exptools](https://github.com/VU-Cog-Sci/exptools2) package.
This is a wrapper around `psychopy` that divides the experiment in several phases (e.g., ITI and stimulus period), which can be extracted.
Similar to the functional files, the output of this is a dataframe indexed by subject/run/task ID:

![plot](https://github.com/gjheij/lazyfmri/blob/main/data/df_onsets.png)

- Eyetracker files: this package is designed to work with the `*.edf`-output from [EyeLink](https://www.sr-research.com/eyelink-1000-plus/) from SR research.
It is required that the system has access to the `edf2asc` command.
The package then uses [hedfpy](https://github.com/tknapen/hedfpy) to parse the eyetracking data into a dataframe indexed by subject/run/task/eye.

### [HRF estimation](https://github.com/gjheij/lazyfmri/blob/main/lazyfmri/fitting.py)

The nature of the output makes the output directly compatible with HRF estimation routines including deconvolution, parameter estimation, and epoching.
Given that all these different functions output similar dataframes, they are all compatible with the `HRFMetrics`-class. 
This extracts a bunch of relevant metrics from your estimated HRF profiles:

![plot](https://github.com/gjheij/lazyfmri/blob/main/data/df_metrics.png)

#### [Deconvolution](https://github.com/gjheij/lazyfmri/blob/main/lazyfmri/fitting.py#L943)

Deconvolution is performed using [nideconv](https://nideconv.readthedocs.io/en/latest/).
Several basis sets are available:

```python
allowed_basis_sets = [
    "fir",      # most flexible, use "regressors='tr'" to use 1 regressor per TR
    "fourier",  # smooth basis set
    "dct",      # smooth basis set without wobbly patterns near overshoot
    "legendre", # never used, really
    "canonical_hrf",    # standard glover HRF
    "canonical_hrf_with_time_derivative",   # standard+time derivative
    "canonical_hrf_with_time_derivative_dispersion" # standard + time/dispersion derivative
]
```

#### [Parameter fitting](https://github.com/gjheij/lazyfmri/blob/main/lazyfmri/fitting.py#L747)

Similar to [pRF](https://github.com/VU-Cog-Sci/prfpy), HRFs can be estimated using an optimizing algorithm.
Starting from the standard HRF parameters, the optimizers searches for the best combination of parameters.
This gives slightly more degrees-of-freedom compared to the `canonical HRF` (with derivatives) while maintaining smooth functions (FIR can go crazy on noisy data such as line-scanning data).
For now, it does the process while trying parameters that generate both a positive and negative HRF, so this can take a long time..

#### [Epoching](https://github.com/gjheij/lazyfmri/blob/main/lazyfmri/fitting.py#L4302)

The OG of HRF estimation.
If you ITIs are long enough, you might be able to extract data from a particular time window around the stimulus onset.
The nice thing about this is that it's very close to the data and doesn't rely on any optimizing algorithm.
On the other hand, the number of trials are inherently limited as they need to be spaced apart far enough to allow the HRF to return to baseline.

### [Plotting](https://github.com/gjheij/lazyfmri/blob/main/lazyfmri/plotting.py)

This package basically represents a wrapper around [seaborn](https://seaborn.pydata.org), which is a wrapper around [matplotlib](https://matplotlib.org).
It goes a few steps further regarding cosmetics such as axis/tick thickness, ratio between font size/title size, shading of error measurements, and much more.
For now, it includes wrappers for simple line plots, bar plots, and correlation plots.
Most notably, it uses [pingouin](https://pingouin-stats.org/build/html/index.html) as statistical backend to estimate significance levels in bar plots and draws significance bars based on the outcome. 
All arguments from `pingouin` can be passed.
It will run non-parametric or parametric tests depending on the variability and spread in your data.

### [Preprocessing](https://github.com/gjheij/lazyfmri/blob/main/lazyfmri/preproc.py)

There are also a bunch of preprocessing functions tailored for the dataframes that are outputted by the `Dataset`-class.
These include highpass (DCT)/lowpass (Savitsky-Golay) filtering, ICA, or power spectra.
Most notably, given a dataframe representing and one representing the onsets, you can regress out certain events from the timecourses (such as `responses` or `blinks`).
