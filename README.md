# dmt-dynamics

Analysis code for:

**Dobbin, E. (2026). Dynamic Neural State Transitions Predict Psychedelic Phenomenal Richness: Magnitude, Not Direction, Drives Subjective Intensity.** Zenodo. https://doi.org/10.5281/zenodo.18167553

---

## What this does

Computes the Consciousness Gradient Index (CGI = √(φ × ρ)) across 30-second windows during DMT EEG recordings, then correlates dynamic measures of CGI change (rate of change, variability, range) with validated phenomenological scores from the 5D-Altered States of Consciousness scale.

Key finding: the *rate* of CGI change predicts phenomenal richness (r = 0.378, p = 0.028); static CGI levels do not.

---

## Data

This script requires the Timmermann et al. (2019) DMT EEG dataset, freely available on Zenodo:

**Dataset DOI:** https://doi.org/10.5281/zenodo.3992359

Download and extract the dataset. You will need:
- The `.bdf` EEG files
- `scales_results.csv` (subjective ratings)

---

## Installation

```bash
pip install mne numpy pandas scipy matplotlib
```

Python 3.9+ recommended.

---

## Usage

Run the script pointing at your local copy of the dataset:

```bash
python cgi_dynamics_analysis.py --data_dir /path/to/dataset --output_dir /path/to/results
```

If you run it from inside the dataset folder with no arguments, it defaults to the current directory:

```bash
cd /path/to/dataset
python cgi_dynamics_analysis.py
```

---

## Outputs

| File | Description |
|------|-------------|
| `cgi_dynamics_results.csv` | Per-subject CGI dynamics and phenomenology scores |
| `cgi_window_timeseries.csv` | Window-by-window CGI values for all subjects |
| `cgi_dynamics_scatter.png` | Scatter plots of CGI metrics vs phenomenal richness |
| `cgi_timeseries_examples.png` | CGI time series for low/medium/high richness participants |

---

## Citation

If you use this code, please cite the paper:

```
Dobbin, E. (2026). Dynamic Neural State Transitions Predict Psychedelic Phenomenal 
Richness: Magnitude, Not Direction, Drives Subjective Intensity. Zenodo. 
https://doi.org/10.5281/zenodo.18167553
```

And the original dataset:

```
Timmermann, C., et al. (2019). Neural and subjective effects of inhaled DMT in 
natural settings. Zenodo. https://doi.org/10.5281/zenodo.3992359
```

---

## Contact

Emma Dobbin — Consciousness Gradient Theory Group Ltd  
contact@cgtheory.com | cgtheory.com
