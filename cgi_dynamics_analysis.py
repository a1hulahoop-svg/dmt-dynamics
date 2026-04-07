import mne
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import coherence, hilbert
import matplotlib.pyplot as plt
import warnings
import os
import argparse
warnings.filterwarnings('ignore')

# Command-line arguments
parser = argparse.ArgumentParser(description='CGI Dynamics Analysis for DMT EEG data')
parser.add_argument('--data_dir', type=str, default='.', help='Path to folder containing .bdf files and scales_results.csv')
parser.add_argument('--output_dir', type=str, default='.', help='Path to folder where results will be saved')
args = parser.parse_args()

# Configuration
DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir
POSTERIOR_CHANNELS = ['O1', 'O2', 'PO3', 'PO4', 'P3', 'P4', 'Pz']
WINDOW_SEC = 30
FMIN, FMAX = 1, 45

# Load subjective ratings
scales = pd.read_csv(f'{DATA_DIR}/scales_results.csv')
scales['Subject'] = scales['Unnamed: 0'].str.upper()

scales['VISUAL_RICHNESS'] = (scales['5D-ComplexImagery'] + scales['5D-ElementaryImagery'] +
                             scales['5D-VR1'] + scales['5D-VR2'] + scales['5D-VR3'])
scales['PHENOMENAL_RICHNESS'] = (scales['VISUAL_RICHNESS'] +
                                  scales['5D-ChangedMeaning'] + scales['5D-Insightfulness'] +
                                  scales['5D-AudioVisualSyn'])

# Lempel-Ziv Complexity
def lempel_ziv_complexity(signal_data, threshold='median'):
    if threshold == 'median':
        thresh = np.median(signal_data)
    else:
        thresh = np.mean(signal_data)

    binary = ''.join(['1' if x > thresh else '0' for x in signal_data])
    n = len(binary)
    if n == 0:
        return 0

    i = 0
    c = 1
    l = 1

    while i + l <= n:
        substring = binary[i:i+l]
        search_space = binary[0:i+l-1]
        if substring in search_space:
            l += 1
        else:
            c += 1
            i += l
            l = 1

    if n > 1:
        max_complexity = n / np.log2(n)
        return c / max_complexity
    return 0

def compute_lzc_multichannel(data, fs):
    n_channels = data.shape[0]
    lzc_values = []
    for ch in range(n_channels):
        analytic_signal = hilbert(data[ch])
        amplitude_envelope = np.abs(analytic_signal)
        downsample_factor = max(1, int(fs / 100))
        envelope_ds = amplitude_envelope[::downsample_factor]
        lzc = lempel_ziv_complexity(envelope_ds)
        lzc_values.append(lzc)
    return np.mean(lzc_values)

def global_coherence(data, fs, fmin=1, fmax=45):
    n_channels = data.shape[0]
    coherences = []
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            freqs, coh = coherence(data[i], data[j], fs=fs, nperseg=min(len(data[0]), int(fs*2)))
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            coherences.append(np.mean(coh[freq_mask]))
    return np.mean(coherences)

def sig_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''

# Get files
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.bdf')]
subjects = sorted(set([f.split('-')[0].upper() if '-' in f else f.split('_')[0].upper() for f in files]))

print('=' * 70)
print('CGI DYNAMICS ANALYSIS')
print('Hypothesis: Richness correlates with DYNAMICS, not averages')
print('=' * 70)
print()

# Store all window-level data
all_windows = []
subject_dynamics = []

for subj in subjects:
    try:
        # Find DMT file
        dmt_file = None
        for f in files:
            if subj.upper() in f.upper() and 'DMT' in f.upper():
                dmt_file = f
                break

        if dmt_file is None:
            continue

        raw = mne.io.read_raw_bdf(f'{DATA_DIR}/{dmt_file}', preload=True, verbose=False)
        raw_filt = raw.copy().filter(FMIN, FMAX, verbose=False)

        fs = raw.info['sfreq']
        ch_names = [ch for ch in raw.ch_names if ch != 'Status']
        posterior_idx = [ch_names.index(ch) for ch in POSTERIOR_CHANNELS if ch in ch_names]

        if len(posterior_idx) < 5:
            continue

        data = raw_filt.get_data()[:-1]
        window_samples = int(WINDOW_SEC * fs)
        n_windows = len(data[0]) // window_samples

        phi_series = []
        rho_series = []
        cgi_series = []

        for w in range(n_windows):
            start = w * window_samples
            end = start + window_samples
            window_data = data[:, start:end]

            phi = global_coherence(window_data, fs, FMIN, FMAX)
            posterior_data = window_data[posterior_idx]
            rho = compute_lzc_multichannel(posterior_data, fs)
            cgi = np.sqrt(phi * rho) * 10

            phi_series.append(phi)
            rho_series.append(rho)
            cgi_series.append(cgi)

            all_windows.append({
                'Subject': subj,
                'Window': w,
                'Time_min': (w * WINDOW_SEC) / 60,
                'Phi': phi,
                'Rho': rho,
                'CGI': cgi
            })

        # Compute dynamics metrics
        phi_arr = np.array(phi_series)
        rho_arr = np.array(rho_series)
        cgi_arr = np.array(cgi_series)

        # Rate of change (first derivative)
        phi_diff = np.diff(phi_arr)
        rho_diff = np.diff(rho_arr)
        cgi_diff = np.diff(cgi_arr)

        dynamics = {
            'Subject': subj,
            'N_Windows': n_windows,

            # Averages
            'Phi_Mean': np.mean(phi_arr),
            'Rho_Mean': np.mean(rho_arr),
            'CGI_Mean': np.mean(cgi_arr),

            # Peaks
            'Phi_Peak': np.max(phi_arr),
            'Rho_Peak': np.max(rho_arr),
            'CGI_Peak': np.max(cgi_arr),

            # Troughs
            'Phi_Min': np.min(phi_arr),
            'Rho_Min': np.min(rho_arr),
            'CGI_Min': np.min(cgi_arr),

            # Range
            'Phi_Range': np.max(phi_arr) - np.min(phi_arr),
            'Rho_Range': np.max(rho_arr) - np.min(rho_arr),
            'CGI_Range': np.max(cgi_arr) - np.min(cgi_arr),

            # Variance/SD (fluctuation)
            'Phi_SD': np.std(phi_arr),
            'Rho_SD': np.std(rho_arr),
            'CGI_SD': np.std(cgi_arr),

            # Coefficient of variation
            'Phi_CV': np.std(phi_arr) / np.mean(phi_arr) if np.mean(phi_arr) > 0 else 0,
            'Rho_CV': np.std(rho_arr) / np.mean(rho_arr) if np.mean(rho_arr) > 0 else 0,
            'CGI_CV': np.std(cgi_arr) / np.mean(cgi_arr) if np.mean(cgi_arr) > 0 else 0,

            # Rate of change metrics
            'Phi_MeanAbsChange': np.mean(np.abs(phi_diff)),
            'Rho_MeanAbsChange': np.mean(np.abs(rho_diff)),
            'CGI_MeanAbsChange': np.mean(np.abs(cgi_diff)),

            # Max rate of change
            'Phi_MaxChange': np.max(np.abs(phi_diff)),
            'Rho_MaxChange': np.max(np.abs(rho_diff)),
            'CGI_MaxChange': np.max(np.abs(cgi_diff)),

            # Trend (slope)
            'Phi_Slope': np.polyfit(range(len(phi_arr)), phi_arr, 1)[0],
            'Rho_Slope': np.polyfit(range(len(rho_arr)), rho_arr, 1)[0],
            'CGI_Slope': np.polyfit(range(len(cgi_arr)), cgi_arr, 1)[0],
        }

        subject_dynamics.append(dynamics)
        print(f'{subj}: {n_windows} windows | CGI: mean={dynamics["CGI_Mean"]:.2f}, peak={dynamics["CGI_Peak"]:.2f}, SD={dynamics["CGI_SD"]:.2f}')

    except Exception as e:
        print(f'{subj}: Error - {str(e)[:40]}')

# Create dataframes
df_windows = pd.DataFrame(all_windows)
df_dynamics = pd.DataFrame(subject_dynamics)

# Merge with subjective ratings
df_merged = df_dynamics.merge(scales[['Subject', '5D-Total', 'VISUAL_RICHNESS', 'PHENOMENAL_RICHNESS']],
                               on='Subject', how='inner')

print()
print('=' * 70)
print(f'DYNAMICS ANALYSIS RESULTS (n={len(df_merged)})')
print('=' * 70)
print()

# Define correlation groups
average_vars = ['Phi_Mean', 'Rho_Mean', 'CGI_Mean']
peak_vars = ['Phi_Peak', 'Rho_Peak', 'CGI_Peak']
fluctuation_vars = ['Phi_SD', 'Rho_SD', 'CGI_SD', 'Phi_CV', 'Rho_CV', 'CGI_CV']
range_vars = ['Phi_Range', 'Rho_Range', 'CGI_Range']
change_vars = ['Phi_MeanAbsChange', 'Rho_MeanAbsChange', 'CGI_MeanAbsChange']
max_change_vars = ['Phi_MaxChange', 'Rho_MaxChange', 'CGI_MaxChange']

subjective = 'PHENOMENAL_RICHNESS'

print('=== CORRELATIONS WITH PHENOMENAL RICHNESS (Spearman) ===')
print()

def print_correlations(var_list, label):
    print(f'--- {label} ---')
    for var in var_list:
        r, p = stats.spearmanr(df_merged[var], df_merged[subjective])
        print(f'  {var:<25} r={r:>7.3f}, p={p:.4f} {sig_stars(p)}')
    print()

print_correlations(average_vars, 'AVERAGE VALUES')
print_correlations(peak_vars, 'PEAK VALUES')
print_correlations(fluctuation_vars, 'FLUCTUATION (SD, CV)')
print_correlations(range_vars, 'RANGE (Max-Min)')
print_correlations(change_vars, 'MEAN RATE OF CHANGE')
print_correlations(max_change_vars, 'MAX RATE OF CHANGE')

# Find best predictor
print('=' * 70)
print('BEST PREDICTORS OF PHENOMENAL RICHNESS')
print('=' * 70)
print()

all_vars = average_vars + peak_vars + fluctuation_vars + range_vars + change_vars + max_change_vars
correlations = []
for var in all_vars:
    r, p = stats.spearmanr(df_merged[var], df_merged[subjective])
    correlations.append({'Variable': var, 'r': r, 'p': p, 'abs_r': abs(r)})

df_corr = pd.DataFrame(correlations).sort_values('abs_r', ascending=False)
print('Top 10 predictors (by |r|):')
print()
for i, row in df_corr.head(10).iterrows():
    print(f"  {row['Variable']:<25} r={row['r']:>7.3f}, p={row['p']:.4f} {sig_stars(row['p'])}")

# Save results
df_merged.to_csv(f'{OUTPUT_DIR}/cgi_dynamics_results.csv', index=False)
df_windows.to_csv(f'{OUTPUT_DIR}/cgi_window_timeseries.csv', index=False)
print()
print(f'Results saved to: {OUTPUT_DIR}/cgi_dynamics_results.csv')
print(f'Window data saved to: {OUTPUT_DIR}/cgi_window_timeseries.csv')

# Generate visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('CGI Dynamics vs Phenomenal Richness', fontsize=14, fontweight='bold')

plot_vars = [
    ('CGI_Mean', 'CGI Mean (Average)'),
    ('CGI_Peak', 'CGI Peak (Maximum)'),
    ('CGI_SD', 'CGI SD (Fluctuation)'),
    ('CGI_Range', 'CGI Range (Max-Min)'),
    ('CGI_MeanAbsChange', 'CGI Mean Rate of Change'),
    ('CGI_CV', 'CGI Coefficient of Variation')
]

for i, (var, label) in enumerate(plot_vars):
    ax = axes[i//3, i%3]
    x = df_merged[var]
    y = df_merged['PHENOMENAL_RICHNESS']

    ax.scatter(x, y, alpha=0.7, s=60, c='steelblue', edgecolors='white', linewidth=0.5)

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(x)
    ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2)

    r, pval = stats.spearmanr(x, y)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('Phenomenal Richness', fontsize=11)
    ax.set_title(f'r={r:.3f}, p={pval:.4f}{sig_stars(pval)}', fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cgi_dynamics_scatter.png', dpi=150, bbox_inches='tight')
print(f'Scatter plots saved to: {OUTPUT_DIR}/cgi_dynamics_scatter.png')
plt.close()

# Time series plot for a few example subjects
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('CGI Time Series During DMT (Example Subjects)', fontsize=14, fontweight='bold')

# Get subjects with high, medium, low richness
df_sorted = df_merged.sort_values('PHENOMENAL_RICHNESS')
example_subjects = [
    df_sorted.iloc[0]['Subject'],  # Low richness
    df_sorted.iloc[len(df_sorted)//2]['Subject'],  # Medium
    df_sorted.iloc[-1]['Subject']  # High richness
]
labels = ['Low Richness', 'Medium Richness', 'High Richness']
colors = ['blue', 'orange', 'green']

for i, (subj, label, color) in enumerate(zip(example_subjects, labels, colors)):
    ax = axes[i]
    subj_data = df_windows[df_windows['Subject'] == subj]
    richness = df_merged[df_merged['Subject'] == subj]['PHENOMENAL_RICHNESS'].values[0]

    ax.plot(subj_data['Time_min'], subj_data['CGI'], '-o', color=color,
            linewidth=2, markersize=6, label=f'{subj}')
    ax.axhline(y=subj_data['CGI'].mean(), color=color, linestyle='--', alpha=0.5, label='Mean')
    ax.fill_between(subj_data['Time_min'], subj_data['CGI'].min(), subj_data['CGI'],
                    alpha=0.2, color=color)

    ax.set_ylabel('CGI', fontsize=11)
    ax.set_title(f'{label}: {subj} (Richness={richness:.0f}, SD={subj_data["CGI"].std():.2f})', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Time (minutes)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cgi_timeseries_examples.png', dpi=150, bbox_inches='tight')
print(f'Time series plots saved to: {OUTPUT_DIR}/cgi_timeseries_examples.png')
plt.close()

# Summary statistics comparison
print()
print('=' * 70)
print('HYPOTHESIS TEST: DYNAMICS vs AVERAGES')
print('=' * 70)
print()

r_mean, p_mean = stats.spearmanr(df_merged['CGI_Mean'], df_merged['PHENOMENAL_RICHNESS'])
r_peak, p_peak = stats.spearmanr(df_merged['CGI_Peak'], df_merged['PHENOMENAL_RICHNESS'])
r_sd, p_sd = stats.spearmanr(df_merged['CGI_SD'], df_merged['PHENOMENAL_RICHNESS'])
r_range, p_range = stats.spearmanr(df_merged['CGI_Range'], df_merged['PHENOMENAL_RICHNESS'])
r_change, p_change = stats.spearmanr(df_merged['CGI_MeanAbsChange'], df_merged['PHENOMENAL_RICHNESS'])

print(f'CGI Mean (average level):     r={r_mean:.3f}, p={p_mean:.4f} {sig_stars(p_mean)}')
print(f'CGI Peak (maximum):           r={r_peak:.3f}, p={p_peak:.4f} {sig_stars(p_peak)}')
print(f'CGI SD (fluctuation):         r={r_sd:.3f}, p={p_sd:.4f} {sig_stars(p_sd)}')
print(f'CGI Range (dynamic range):    r={r_range:.3f}, p={r_range:.4f} {sig_stars(p_range)}')
print(f'CGI Rate of Change:           r={r_change:.3f}, p={p_change:.4f} {sig_stars(p_change)}')
print()

# Determine if dynamics are better predictors
dynamics_rs = [abs(r_peak), abs(r_sd), abs(r_range), abs(r_change)]
if max(dynamics_rs) > abs(r_mean):
    best_dynamic = ['Peak', 'SD', 'Range', 'Rate'][dynamics_rs.index(max(dynamics_rs))]
    print(f'RESULT: {best_dynamic} (|r|={max(dynamics_rs):.3f}) is a better predictor than Mean (|r|={abs(r_mean):.3f})')
else:
    print(f'RESULT: Mean (|r|={abs(r_mean):.3f}) is the best predictor')
