#!/usr/bin/env python3
## Run multiple jobs in parallel (array job)

"""
Collect and average fidelity results from multiple runs.
"""

import numpy as np
import glob
from pathlib import Path

# Configuration
RUN_NAME = 'default_fidelity_run'  # Change this to match your run
RESULTS_DIR = Path(f'./runs_fidelity/{RUN_NAME}/fidelity_results')

# Find all result files
result_files = sorted(RESULTS_DIR.glob('run_*.npz'))
print(f"Found {len(result_files)} result files")

if len(result_files) == 0:
    print(f"No results found in {RESULTS_DIR}")
    sys.exit(1)

# Load first file to get structure
first_data = np.load(result_files[0])
noise_levels = first_data['noise_levels']
param_names = [key.replace('fidelities_', '') 
               for key in first_data.keys() if key.startswith('fidelities_')]

# Initialize storage
all_fidelities = {name: [] for name in param_names}

# Collect all data
for file_path in result_files:
    data = np.load(file_path)
    for name in param_names:
        all_fidelities[name].append(data[f'fidelities_{name}'])

# Convert to arrays
for name in param_names:
    all_fidelities[name] = np.array(all_fidelities[name])  # shape: (n_runs, n_noise_levels)

# Calculate statistics
results_summary = {
    'noise_levels': noise_levels,
    'parameters': {}
}

for name in param_names:
    results_summary['parameters'][name] = {
        'mean': np.mean(all_fidelities[name], axis=0),
        'std': np.std(all_fidelities[name], axis=0),
        'all_runs': all_fidelities[name]
    }

# Save summary
summary_path = RESULTS_DIR.parent / 'fidelity_summary.npz'
np.savez(
    summary_path,
    noise_levels=noise_levels,
    **{f'{name}_mean': results_summary['parameters'][name]['mean'] 
       for name in param_names},
    **{f'{name}_std': results_summary['parameters'][name]['std'] 
       for name in param_names}
)

# Print summary
print("\n" + "="*60)
print("FIDELITY SUMMARY")
print("="*60)
print(f"Number of runs: {len(result_files)}")
print(f"Noise levels: {len(noise_levels)}")
print("\nAverage fidelities at highest noise level:")

for name in param_names:
    mean_last = results_summary['parameters'][name]['mean'][-1]
    std_last = results_summary['parameters'][name]['std'][-1]
    print(f"  {name}: {mean_last:.4f} ± {std_last:.4f}")

print(f"\nSummary saved to: {summary_path}")

# Optional: Create a simple plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for name in param_names:
    mean = results_summary['parameters'][name]['mean']
    std = results_summary['parameters'][name]['std']
    
    plt.plot(noise_levels, mean, 'o-', label=name, linewidth=2)
    plt.fill_between(noise_levels, mean-std, mean+std, alpha=0.2)

plt.xlabel('Noise Level (std dev)')
plt.ylabel('Fidelity')
plt.title(f'Fidelity vs Noise ({len(result_files)} runs)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = RESULTS_DIR.parent / 'fidelity_vs_noise.png'
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to: {plot_path}")