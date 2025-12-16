"""
Main training script for Hamiltonian parameter inference from STS data.
Run with environment variables to configure training.
Example: RUN_NAME=my_exp NCOMP=300 BATCH=128 python train.py
"""

import os
import sys
import numpy as np
import helper as hp
from config import get_config
from data_loader import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import matplotlib.pyplot as plt

def build_mlp(in_dim, width=(384, 256, 128), dropout=(0.15, 0.10, 0.0), l2w=2e-4):
    """Build Multi-Layer Perceptron for parameter regression."""
    x = tf.keras.Input(shape=(in_dim,), name='input')
    h = x
    for i, units in enumerate(width):
        h = layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(l2w))(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)
        if i < len(dropout) and dropout[i] > 0:
            h = layers.Dropout(dropout[i])(h)
    out = layers.Dense(1, activation='linear', name='output')(h)
    return Model(inputs=x, outputs=out)

def train_model(model_name, X_train, X_val, y_train, y_val, config, scaler, 
                X_test=None, y_test=None):
    """Train a single model for one Hamiltonian parameter."""
    
    print(f"\n=== Training model for {model_name} ===")
    
    # Build model
    model = build_mlp(
        X_train.shape[1], 
        width=config.MODEL_WIDTHS.get(model_name, config.DEFAULT_WIDTH),
        dropout=config.DROPOUT_RATES.get(model_name, config.DEFAULT_DROPOUT),
        l2w=config.L2_REGULARIZATION
    )
    
    # Compile
    loss_fn = tf.keras.losses.Huber(delta=config.LOSS_DELTAS.get(model_name, 0.1))
    opt = optimizers.Adam(learning_rate=config.LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss_fn)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            f"{config.OUTPUT_DIR}/model_{model_name}/best.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Save
    model.save(f"{config.OUTPUT_DIR}/model_{model_name}/final.keras")
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_true = scaler.inverse_transform(y_test)
        fid = hp.fidelity(y_true, y_pred)
        
        # Generate diagnostic plots
        hp.plot_training_history(history, model_name, config.OUTPUT_DIR)
        hp.plot_parity_plot(y_true, y_pred, model_name, fid, config.OUTPUT_DIR)
        
        return fid, y_true, y_pred
    
    return None, None, None

def main():
    """Main training pipeline."""
    
    # Load configuration
    config = get_config()
    print(f"Starting training with configuration:")
    print(f"  Run name: {config.RUN_NAME}")
    print(f"  PCA components: {config.N_COMPONENTS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.EPOCHS}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    X, labels, metadata = load_and_preprocess_data(
        data_paths=['{directory1}/10k_dataset.npz', '{directory2}/2k_dataset_narrow_windows.npz'], # download datasets from Zenodo or generate your own
        num_freq=251,
        num_eps=80
    )
    
    # Split data
    X_train, X_test, labels_train, labels_test = train_test_split(
        X, labels, test_size=0.2, random_state=config.RANDOM_SEED, shuffle=True
    )
    
    X_train, X_val, labels_train, labels_val = train_test_split(
        X_train, labels_train, test_size=0.15, random_state=config.RANDOM_SEED, shuffle=True
    )
    
    # Scale targets
    scalers = {}
    scaled_labels = {}
    
    param_names = ['tau', 'soc', 'emin', 'emax']
    for i, name in enumerate(param_names):
        scaler = StandardScaler()
        scalers[name] = scaler
        scaled_labels[f'{name}_train'] = scaler.fit_transform(labels_train[:, i].reshape(-1, 1))
        scaled_labels[f'{name}_val'] = scaler.transform(labels_val[:, i].reshape(-1, 1))
        scaled_labels[f'{name}_test'] = scaler.transform(labels_test[:, i].reshape(-1, 1))

    print("Adding Gaussian noise to the training set...")
    X_train_noisy = hp.add_noise(X_train, std_dev=0.02)
    # Apply PCA
    print("Applying PCA for dimensionality reduction...")

    pca = PCA(n_components=config.N_COMPONENTS)
    pca.fit(X_train)                    # fit PCA on clean data
    X_train_pca = pca.transform(X_train_noisy)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    # Save PCA model
    joblib.dump(pca, f"{config.OUTPUT_DIR}/pca_model.pkl")
    
    # Train models for each parameter
    results = {}
    for name in param_names:
        fid, y_true, y_pred = train_model(
            name,
            X_train_pca,
            X_val_pca,
            scaled_labels[f'{name}_train'],
            scaled_labels[f'{name}_val'],
            config,
            scalers[name],
            X_test_pca,
            scaled_labels[f'{name}_test']
        )
        results[name] = fid
        joblib.dump(scalers[name], f"{config.OUTPUT_DIR}/model_{name}/scaler.pkl")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for name, fid in results.items():
        print(f"{name}: Fidelity = {fid:.4f}")
    
    # Save final summary
    with open(f"{config.OUTPUT_DIR}/training_summary.txt", "w") as f:
        f.write(f"Run: {config.RUN_NAME}\n")
        f.write(f"Date: {config.timestamp}\n")
        f.write(f"PCA components: {config.N_COMPONENTS}\n")
        f.write(f"Random features: {config.N_RANDOM_FEATURES}\n")
        f.write(f"Test fidelities:\n")
        for name, fid in results.items():
            f.write(f"  {name}: {fid:.4f}\n")

if __name__ == "__main__":
    main()