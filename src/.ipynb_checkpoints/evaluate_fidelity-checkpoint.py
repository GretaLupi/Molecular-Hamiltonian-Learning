#!/usr/bin/env python3
"""
Evaluate fidelity vs noise level for Hamiltonian parameter inference.
Run with array jobs: python evaluate_fidelity.py <repeat_index>
"""

import os
import sys
import numpy as np
import joblib
from pathlib import Path

# Add helper module
sys.path.append('.')
import helper as hp

# Set random seeds
repeat_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
np.random.seed(1234 + repeat_index)

# ===== CONFIGURATION =====
# Can override with environment variables
RUN_NAME = os.getenv('RUN_NAME', 'default_fidelity_run')
DATA_DIR = os.getenv('DATA_DIR', './data')
OUTPUT_DIR = Path(f'./runs_fidelity/{RUN_NAME}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
N_COMPONENTS = int(os.getenv('NCOMP', 200))
EPOCHS = int(os.getenv('EPOCHS', 600))
BATCH_SIZE = int(os.getenv('BATCH', 64))
LEARNING_RATE = float(os.getenv('LR', 2e-4))
PATIENCE = int(os.getenv('PATIENCE', 200))

# Noise levels to test
NOISE_LEVELS = np.linspace(0.0, 1.0, 10)  # 0% to 100% noise

print(f"Starting fidelity evaluation run {repeat_index}")
print(f"Configuration: N_COMP={N_COMPONENTS}, EPOCHS={EPOCHS}")
print(f"Noise levels: {len(NOISE_LEVELS)} from {NOISE_LEVELS[0]} to {NOISE_LEVELS[-1]}")

# ===== LOAD DATA =====
print("\nLoading data...")
data_files = list(Path(DATA_DIR).glob('*.npz'))
if not data_files:
    print(f"Error: No .npz files found in {DATA_DIR}")
    sys.exit(1)

# Load and preprocess (assuming you have data_loader module)
from data_loader import load_and_preprocess_data

X, labels, metadata = load_and_preprocess_data(
    data_paths=data_files,
    num_freq=251,
    num_eps=80
)

print(f"Loaded {len(X)} samples with {X.shape[1]} features")

# ===== TRAIN-TEST SPLIT =====
print("\nSplitting data...")
from sklearn.model_selection import train_test_split

# Split: 70% train, 15% val, 15% test
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42, shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15/0.7, random_state=42, shuffle=True
)  # 0.15/0.7 ≈ 0.214 to get 15% of total as val

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ===== SCALE TARGETS =====
print("\nScaling targets...")
from sklearn.preprocessing import StandardScaler

scalers = {}
scaled_y = {}

param_names = ['tau', 'soc', 'emin', 'emax']
for i, name in enumerate(param_names):
    scaler = StandardScaler()
    scalers[name] = scaler
    scaled_y[f'{name}_train'] = scaler.fit_transform(y_train[:, i:i+1])
    scaled_y[f'{name}_val'] = scaler.transform(y_val[:, i:i+1])
    scaled_y[f'{name}_test'] = scaler.transform(y_test[:, i:i+1])

# ===== PCA =====
print("\nApplying PCA...")
from sklearn.decomposition import PCA

# Add noise to training data for regularization
X_train_noisy = hp.add_noise(X_train, std_dev=0.02)

# Fit PCA on noisy training data
pca = PCA(n_components=N_COMPONENTS)
pca.fit(X_train)

# Transform all sets
X_train_pca = pca.transform(X_train_noisy)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# Save PCA model
pca_path = OUTPUT_DIR / 'pca_model.pkl'
joblib.dump(pca, pca_path)

# ===== TRAIN MODELS =====
print("\nTraining models...")
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

tf.random.set_seed(5678 + repeat_index)

def build_model(input_dim):
    """Simple MLP model."""
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    return model

models = {}
for name in param_names:
    print(f"  Training {name} model...")
    model = build_model(X_train_pca.shape[1])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse'  # Simple MSE for all parameters
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(
            OUTPUT_DIR / f'model_{name}.keras',
            save_best_only=True
        )
    ]
    
    # Train
    history = model.fit(
        X_train_pca,
        scaled_y[f'{name}_train'],
        validation_data=(X_val_pca, scaled_y[f'{name}_val']),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )
    
    models[name] = model
    print(f"    Finished with val_loss: {history.history['val_loss'][-1]:.4f}")

# ===== EVALUATE FIDELITY VS NOISE =====
print(f"\nEvaluating fidelity vs noise (run {repeat_index})...")

# Store results
results = {
    'noise_levels': NOISE_LEVELS,
    'fidelities': {}
}

for name in param_names:
    results['fidelities'][name] = np.zeros(len(NOISE_LEVELS))

# True values (unscaled)
true_values = {}
for i, name in enumerate(param_names):
    true_values[name] = y_test[:, i:i+1]

# Test for each noise level
for i, noise_level in enumerate(NOISE_LEVELS):
    print(f"  Noise level {i+1}/{len(NOISE_LEVELS)}: {noise_level:.3f}")
    
    # Add noise to test data
    X_test_noisy = hp.add_noise(X_test, std_dev=noise_level)
    X_test_pca_noisy = pca.transform(X_test_noisy)
    
    # Predict for each parameter
    for name in param_names:
        # Get predictions
        y_pred_scaled = models[name].predict(X_test_pca_noisy, verbose=0)
        y_pred = scalers[name].inverse_transform(y_pred_scaled)
        
        # Calculate fidelity
        fid = hp.fidelity(true_values[name], y_pred)
        results['fidelities'][name][i] = fid

# ===== SAVE RESULTS =====
print("\nSaving results...")
save_dir = OUTPUT_DIR / 'fidelity_results'
save_dir.mkdir(exist_ok=True)

# Save this run's results
np.savez(
    save_dir / f'run_{repeat_index:03d}.npz',
    noise_levels=results['noise_levels'],
    **{f'fidelities_{name}': results['fidelities'][name] for name in param_names}
)

# Save scalers for future use
for name in param_names:
    joblib.dump(scalers[name], OUTPUT_DIR / f'scaler_{name}.pkl')

print(f"\nDone! Results saved to {save_dir}/run_{repeat_index:03d}.npz")