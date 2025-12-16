import numpy as np
import helper as hp

def load_and_preprocess_data(data_paths, num_freq=251, num_eps=80):
    """Load and preprocess data from multiple NPZ files."""
    
    all_X_processed = []
    all_labels = []
    all_processing_info = []
    
    for path in data_paths:
        data = np.load(path)
        x = data['x']
        y = data['y']
        labels = data['labels']
        
        freq_axis_ref = None
        if x is not None and len(x) > 0:
            # Take first sample, first epsilon's frequency axis
            freq_axis_ref = x[0, :2*num_freq]
            print(f"  Frequency axis range: [{freq_axis_ref[0]:.3f}, "
                  f"{freq_axis_ref[-1]:.3f}] eV")
            
        # Compute integrated spectra
        print("  Computing integrated spectra...")
        y_integral = hp.compute_integral_array(x, y, num_freq=num_freq, num_eps=num_eps)
        
        # Apply preprocessing
        print("  Applying preprocessing pipeline...")
        # Apply preprocessing pipeline
        X_processed, processing_info = preprocess_spectra(
            y_integral=y_integral,
            x_freq=freq_axis_ref,
            num_freq=num_freq,
            num_eps=num_eps
        )
        
        # Store results
        all_X_processed.append(X_processed)
        all_labels.append(labels_batch)
        all_processing_info.append(processing_info)
        
        print(f"  ✓ Processed: {len(X_processed)} samples, "
              f"{X_processed.shape[1]} features")
    
    # [Combine and return as before...]
    
    # Include processing info in metadata
    metadata['processing_steps'] = all_processing_info
    
    return X, labels, metadata

def preprocess_spectra(y_integral, x_freq=None, num_freq=251, num_eps=80):
    """
    Apply full preprocessing pipeline to integrated spectra.
    
    Steps:
    1. Remove duplicate ω=0 point
    2. Remove central bias region (±20mV)
    3. Remove DC offset per epsilon curve
    4. Normalize per sample
    """
    
    n_samples = y_integral.shape[0]
    
    # ===== 0. EXTRACT OR CREATE FREQUENCY AXIS =====
    if x_freq is None:
        # Create default frequency axis if not provided
        # Assuming symmetric around 0 with duplicate zero
        freq_axis_full = np.linspace(-0.2, 0.2, 2*num_freq)
    else:
        # Use provided frequency axis (take first sample as reference)
        freq_axis_full = x_freq[0, :2*num_freq]  # First sample's frequency axis
    
    # ===== 1. REMOVE DUPLICATE ω=0 POINT =====
    print("  Step 1: Removing duplicate ω=0 point...")
    
    # Create mask to identify which frequency point to drop
    keep_mask = hp.make_keep_mask_for_omega(freq_axis_full)
    
    # Get new frequency axis after removal
    freq_axis = freq_axis_full[keep_mask]
    new_num_freq = len(freq_axis) // 2  # Should be (2*num_freq - 1) total points
    
    print(f"    Original frequencies: {len(freq_axis_full)} points")
    print(f"    After removal: {len(freq_axis)} points")
    print(f"    Removed point at ω = {freq_axis_full[~keep_mask][0]:.6f}")
    
    # Apply removal to data
    y_dropped = hp.drop_omega_idx_in_flat(y_integral, num_eps, keep_mask)
    
    # Reshape to 2D: (samples, epsilon, bias)
    X_2d = y_dropped.reshape(n_samples, num_eps, -1)
    actual_freq_points = X_2d.shape[2]
    
    print(f"    Data shape after ω=0 removal: {X_2d.shape}")
    
    # ===== 2. REMOVE CENTRAL BIAS REGION (±20mV) =====
    print("  Step 2: Removing central bias region (±20mV)...")
    
    # Create bias axis from remaining frequencies
    # Note: freq_axis is already without the duplicate zero
    bias_axis = freq_axis
    
    # Define region to remove (±20mV around 0)
    bias_remove = (bias_axis > -0.03) & (bias_axis < 0.03)
    bias_keep = ~bias_remove
    
    # Indices for negative and positive bias (outside central region)
    neg_indices = np.where(bias_axis < -0.03)[0]
    pos_indices = np.where(bias_axis > 0.03)[0]
    
    print(f"    Bias points: total={len(bias_axis)}, "
          f"keeping {np.sum(bias_keep)} (removing {np.sum(bias_remove)})")
    
    # Remove central region and concatenate
    X_no_central = np.zeros((n_samples, num_eps, len(neg_indices) + len(pos_indices)))
    
    for i in range(n_samples):
        # Take negative and positive bias regions
        neg_part = X_2d[i, :, neg_indices]  # shape: (num_eps, num_neg)
        pos_part = X_2d[i, :, pos_indices]  # shape: (num_eps, num_pos)
        
        # Concatenate along bias axis
        X_no_central[i, :, :] = np.concatenate([neg_part.T, pos_part.T], axis=1)
    
    # ===== 3. REMOVE DC OFFSET =====
    print("  Step 3: Removing DC offset...")
    
    # Subtract value at middle bias point (typically near 0 after processing)
    mid_idx = X_no_central.shape[2] // 2
    X_no_dc = X_no_central - X_no_central[:, :, mid_idx:mid_idx+1]
    
    # ===== 4. NORMALIZE PER SAMPLE =====
    print("  Step 4: Normalizing per sample...")
    
    # Max-normalize each sample independently
    max_vals = np.max(np.abs(X_no_dc), axis=(1, 2), keepdims=True)
    X_normalized = X_no_dc / (max_vals + 1e-12)
    
    # ===== 5. FLATTEN FOR PCA =====
    X_flat = X_normalized.reshape(n_samples, -1)
    
    print(f"  Final shape: {X_flat.shape} "
          f"({num_eps} ε × {X_no_central.shape[2]} ω points)")
    
    # Store processing metadata
    processing_info = {
        'original_freq_points': 2 * num_freq,
        'after_omega0_removal': actual_freq_points,
        'after_bias_removal': X_no_central.shape[2],
        'bias_range_removed': '±30meV',
        'duplicate_zero_removed': True,
        'dc_offset_removed': True,
        'normalization': 'per-sample max'
    }
    
    return X_flat, processing_info