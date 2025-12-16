#!/usr/bin/env python3
"""
Preprocessing pipeline for experimental STS data.
Loads raw CSV, applies smoothing, fitting, and prepares for neural network.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ===== 1. LOADING RAW DATA =====
def load_experimental_data(csv_path):
    """
    Load experimental dI/dV data from CSV.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        df: DataFrame with columns:
            - filename: measurement filename
            - Bias_V: bias voltage (V)
            - Current_A: measured current (A)
            - dIdV_S: raw dI/dV (S)
            - Z_m: tip height (m)
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Sort and add useful columns
    df = df.sort_values(["Current_A", "Bias_V"]).reset_index(drop=True)
    df["I_set_A"] = df.groupby("filename")["Current_A"].transform("first")
    df["I_set_pA"] = df["I_set_A"] * 1e12  # Convert to pA
    df["Current_pA"] = df["Current_A"] * 1e12
    
    print(f"Loaded {len(df)} measurements from {df['filename'].nunique()} files")
    return df

# ===== 2. SAVITZKY-GOLAY SMOOTHING =====
def _odd_below(n):
    """Return odd integer <= n (minimum 3)."""
    k = int(n)
    if k % 2 == 0:
        k -= 1
    return max(3, k)

def apply_savgol_smoothing(df, window_points=21, polyorder=3):
    """
    Apply Savitzky-Golay filter to each measurement file.
    
    Args:
        df: DataFrame with raw data
        window_points: Filter window size
        polyorder: Polynomial order
        
    Returns:
        df_smoothed: DataFrame with added 'dIdV_S_sg' column
    """
    print("Applying Savitzky-Golay smoothing...")
    
    df_smoothed = df.copy()
    smoothed_data = []
    
    for fname, group in df_smoothed.groupby("filename"):
        group = group.sort_values("Bias_V").copy()
        n_points = len(group)
        
        if n_points < 7:
            # Too few points, keep as is
            group["dIdV_S_sg"] = group["dIdV_S"]
        else:
            # Calculate appropriate window size
            w = _odd_below(min(window_points, n_points - (1 if (n_points - polyorder) % 2 == 1 else 2)))
            if w <= polyorder:
                w = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
                w = min(w, _odd_below(n_points))
            
            # Apply filter
            group["dIdV_S_sg"] = savgol_filter(
                group["dIdV_S"].values, 
                window_length=w, 
                polyorder=polyorder, 
                mode="interp"
            )
        
        smoothed_data.append(group)
    
    df_smoothed = pd.concat(smoothed_data, ignore_index=True)
    print(f"Applied smoothing with window={window_points}, polyorder={polyorder}")
    return df_smoothed

# ===== 3. INTERPOLATION TO UNIFORM GRID =====
def interpolate_to_uniform_grid(df, num_eps=80):
    """
    Interpolate data from non-uniform tip heights to uniform grid.
    
    Args:
        df: DataFrame with smoothed data
        num_eps: Number of uniform epsilon (tip-height) points
        
    Returns:
        bias_axis: Bias voltage axis (V)
        eps_axis: Uniform tip-height axis (pA)
        didv_matrix: Interpolated dI/dV matrix (bias × epsilon)
    """
    print(f"Interpolating to {num_eps} uniform epsilon points...")
    
    # Create pivot table
    piv = (df.groupby(["Bias_V", "I_set_pA"])["dIdV_S_sg"]
             .mean().unstack("I_set_pA"))
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    
    # Extract axes
    bias_axis = piv.index.values.astype(float)
    eps_axis_raw = piv.columns.values.astype(float)  # Non-uniform tip heights
    didv_raw = piv.values  # Shape: (bias_points, eps_points_raw)
    
    # Create uniform epsilon axis
    eps_axis_uniform = np.linspace(eps_axis_raw.min(), eps_axis_raw.max(), num_eps)
    
    # Interpolate along epsilon dimension
    didv_interp = np.zeros((len(bias_axis), num_eps))
    for i in range(len(bias_axis)):
        didv_interp[i, :] = np.interp(
            eps_axis_uniform, 
            eps_axis_raw, 
            didv_raw[i, :]
        )
    
    print(f"Interpolated matrix shape: {didv_interp.shape}")
    return bias_axis, eps_axis_uniform, didv_interp

# ===== 4. STEP FUNCTION FITTING =====
import jax
import jax.numpy as jnp
from jax import jit, grad
from scipy.optimize import minimize

# JAX functions for step fitting
def steps_jax(ps, xs):
    """Step function for fitting dI/dV features."""
    n = len(ps) // 3
    locs = jnp.abs(ps[0:n])
    hes = jnp.abs(ps[n:2*n])
    wis = jnp.abs(ps[2*n:3*n])
    
    ys = xs * 0.
    for i in range(n):
        ys = ys + hes[i] * (jnp.tanh(1./wis[i] * (xs - locs[i])) + 1.) / 2.
    return ys

def error_jax(ps, xs, ys0):
    """Error function for step fitting."""
    ys = steps_jax(ps, xs)
    diff = ys0 - ys
    return jnp.sum(diff * diff)

# Compile JAX functions
steps = jit(steps_jax)
error = jit(error_jax)
jac_error = jit(grad(error_jax, argnums=0))

def fit_step_function(xs, ys, maxsteps=8):
    """Fit step function to a single dI/dV curve."""
    def fun(ps):
        return error(ps, xs, ys)
    
    def jac_fun(ps):
        return jac_error(ps, xs, ys)
    
    ntries = 20
    results = []
    
    for i in range(ntries):
        ps0 = np.random.random(3 * maxsteps)
        result = minimize(fun, ps0, jac=jac_fun, tol=1e-6)
        results.append(result)
    
    losses = [fun(result.x) for result in results]
    return results[np.argmin(losses)].x

def fit_didv_curve(didv_curve, bias_axis, nsteps_pos=3, nsteps_neg=3):
    """
    Fit step function to positive and negative bias separately.
    
    Args:
        didv_curve: Single dI/dV curve
        bias_axis: Bias voltage axis
        nsteps_pos: Steps for positive bias
        nsteps_neg: Steps for negative bias
        
    Returns:
        didv_fit: Fitted curve
    """
    # Split positive and negative bias
    mask_neg = bias_axis < 0
    mask_pos = bias_axis >= 0
    
    V_neg = bias_axis[mask_neg]
    V_pos = bias_axis[mask_pos]
    
    Y_neg = didv_curve[mask_neg]
    Y_pos = didv_curve[mask_pos]
    
    # Fit negative part (use absolute bias)
    V_neg_abs = -V_neg[::-1]
    Y_neg_rev = Y_neg[::-1]
    
    ps_neg = fit_step_function(V_neg_abs, Y_neg_rev, maxsteps=nsteps_neg)
    Y_fit_neg_abs = steps(ps_neg, V_neg_abs)
    Y_fit_neg = Y_fit_neg_abs[::-1]
    
    # Fit positive part
    ps_pos = fit_step_function(V_pos, Y_pos, maxsteps=nsteps_pos)
    Y_fit_pos = steps(ps_pos, V_pos)
    
    # Combine
    didv_fit = np.zeros_like(didv_curve)
    didv_fit[mask_neg] = Y_fit_neg
    didv_fit[mask_pos] = Y_fit_pos
    
    return didv_fit

def fit_all_curves(didv_matrix, bias_axis, nsteps_pos=3, nsteps_neg=3):
    """
    Fit step functions to all epsilon curves.
    
    Args:
        didv_matrix: dI/dV matrix (bias × epsilon)
        bias_axis: Bias voltage axis
        
    Returns:
        didv_fitted: Fitted dI/dV matrix
    """
    print("Fitting step functions to all curves...")
    num_eps = didv_matrix.shape[1]
    didv_fitted = np.zeros_like(didv_matrix)
    
    # Remove DC offset for each curve
    didv_dc_removed = didv_matrix - didv_matrix[:, bias_axis == 0]
    
    for eps_idx in range(num_eps):
        didv_fitted[:, eps_idx] = fit_didv_curve(
            didv_dc_removed[:, eps_idx],
            bias_axis,
            nsteps_pos=nsteps_pos,
            nsteps_neg=nsteps_neg
        )
        
        if (eps_idx + 1) % 20 == 0:
            print(f"  Fitted {eps_idx + 1}/{num_eps} curves")
    
    return didv_fitted

# ===== 5. REMOVE CENTRAL BIAS REGION =====
def remove_central_bias(didv_matrix, bias_axis, bound=0.03):
    """
    Remove central bias region (±bound V).
    
    Args:
        didv_matrix: dI/dV matrix (bias × epsilon)
        bias_axis: Bias voltage axis
        bound: Central region bound in volts
        
    Returns:
        didv_processed: Processed matrix without central bias
    """
    print(f"Removing central bias region (±{bound} V)...")
    
    # Find indices to keep
    neg_indices = np.where(bias_axis < -bound)[0]
    pos_indices = np.where(bias_axis > bound)[0]
    
    # Remove central region
    neg_part = didv_matrix[neg_indices, :]  # Shape: (neg_points, eps_points)
    pos_part = didv_matrix[pos_indices, :]  # Shape: (pos_points, eps_points)
    
    # Concatenate along bias axis
    didv_processed = np.concatenate([neg_part, pos_part], axis=0)
    
    print(f"  Original shape: {didv_matrix.shape}")
    print(f"  After removal: {didv_processed.shape}")
    print(f"  Removed {len(bias_axis) - didv_processed.shape[0]} bias points")
    
    return didv_processed


# ===== 6. MAIN PIPELINE =====
def preprocess_experimental_pipeline(csv_path, output_dir="./processed_data"):
    """
    Complete preprocessing pipeline for experimental data.
    
    Args:
        csv_path: Path to raw CSV data
        output_dir: Directory to save processed data
        
    Returns:
        processed_data: Ready-to-use data for neural network
        metadata: Processing metadata
    """

    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data
    df = load_experimental_data(csv_path)
    
    # Step 2: Smoothing
    df_smooth = apply_savgol_smoothing(df)
    
    # Step 3: Interpolation to uniform grid
    bias_axis, eps_axis, didv_interp = interpolate_to_uniform_grid(df_smooth, num_eps=80)
    
    # Step 4: NORMALIZE PER EPSILON (subito dopo interpolazione)
    print("Normalizing per epsilon after interpolation...")
    didv_normalized = didv_interp / (np.max(np.abs(didv_interp), axis=0, keepdims=True) + 1e-12)
    
    # Step 5: Step function fitting (su dati normalizzati)
    print("Fitting step functions to normalized curves...")
    didv_fitted = fit_all_curves(didv_normalized.T, bias_axis*100)  # Transpose to (eps × bias)
    didv_fitted = didv_fitted.T  # Back to (bias × eps)
    
    # Step 6: Remove DC offset (subtract value at bias=0)
    zero_idx = np.argmin(np.abs(bias_axis))
    didv_dc_removed = didv_fitted - didv_fitted[zero_idx:zero_idx+1, :]
    
    # Step 7: Remove central bias region
    didv_no_center = remove_central_bias(didv_dc_removed, bias_axis, bound=0.03)
    
    # Step 8: Transpose to (epsilon × bias) format
    didv_final = didv_no_center.T  # Shape: (80, remaining_bias_points)    
    # Create metadata
    metadata = {
        'original_csv': csv_path,
        'bias_axis_original': bias_axis,
        'eps_axis': eps_axis,
        'final_shape': didv_final.shape,
        'bias_points_removed': len(bias_axis) - didv_normalized.shape[0],
        'processing_steps': [
            'Savitzky-Golay smoothing',
            'Uniform epsilon interpolation',
            'Step function fitting',
            'DC offset removal',
            'Central bias removal (±30mV)',
            'Per-epsilon normalization'
        ]
    }
    
    # Save results
    output_path = os.path.join(output_dir, "experimental_processed.npz")
    np.savez(
        output_path,
        didv_data=didv_final,
        metadata=metadata
    )
    
    print(f"\nPreprocessing complete!")
    print(f"Final data shape: {didv_final.shape}")
    print(f"Saved to: {output_path}")
    
    return didv_final, metadata

# ===== 7. QUICK VISUALIZATION =====
def plot_processed_data(didv_data, output_path="./processed_data/plot.png"):
    """
    Quick visualization of processed data.
    
    Args:
        didv_data: Processed dI/dV matrix (epsilon × bias)
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(didv_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='dI/dV (norm)')
    plt.xlabel('Bias index')
    plt.ylabel('Epsilon index')
    plt.title('Processed dI/dV Map')
    
    plt.subplot(2, 2, 2)
    # Plot first, middle, and last epsilon curves
    eps_indices = [0, didv_data.shape[0]//2, -1]
    for idx in eps_indices:
        plt.plot(didv_data[idx, :], label=f'Epsilon {idx}')
    plt.xlabel('Bias index')
    plt.ylabel('dI/dV (norm)')
    plt.title('Sample Epsilon Curves')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    # Plot bias profiles at different epsilon
    bias_indices = [0, didv_data.shape[1]//2, -1]
    for idx in bias_indices:
        plt.plot(didv_data[:, idx], label=f'Bias {idx}')
    plt.xlabel('Epsilon index')
    plt.ylabel('dI/dV (norm)')
    plt.title('Sample Bias Profiles')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Statistics
    plt.hist(didv_data.flatten(), bins=50, alpha=0.7)
    plt.xlabel('dI/dV value')
    plt.ylabel('Frequency')
    plt.title('Value Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")

# ===== 8. MAIN FUNCTION =====
if __name__ == "__main__":
    # Example usage
    CSV_PATH = "./Molecule_5/didv_combined.csv"  # Change this to your data path
    OUTPUT_DIR = "./processed_experimental"
    
    # Run complete pipeline
    didv_processed, metadata = preprocess_experimental_pipeline(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Create visualization
    plot_processed_data(
        didv_processed,
        output_path=f"{OUTPUT_DIR}/processed_data_plot.png"
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Input CSV: {CSV_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Final data shape: {didv_processed.shape}")
    print(f"Epsilon points: {didv_processed.shape[0]}")
    print(f"Bias points: {didv_processed.shape[1]}")
    
    # Save as flattened array for easy loading
    flattened_path = f"{OUTPUT_DIR}/experimental_flattened.csv"
    np.savetxt(flattened_path, didv_processed.flatten().reshape(1, -1), delimiter=',')
    print(f"\nFlattened data saved to: {flattened_path}")
    print("Ready for PCA and neural network inference!")