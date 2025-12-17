# Molecular Hamiltonian Learning from STM-IETS

Machine learning to invert STM spectra into molecular Hamiltonians. Code for "Molecular Hamiltonian learning from setpoint-dependent scanning tunneling
spectroscopy" (G.Lupi et al., 2025).

## Features
- Synthetic STM-IETS spectrum generation
- Neural networks for Hamiltonian parameter extraction
- Preprocessing tools for experimental data
- Datasets available in Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17950922.svg)](https://doi.org/10.5281/zenodo.17950922)

## Repository Structure

### src/ 
- **`collect_fidelity_results.py`** - Collects and aggregates fidelity results from multiple runs
- **`config.py`** - Configuration settings and parameters for the training
- **`data_loader.py`** - Handles loading and preprocessing of synthetic datasets
- **`evaluate_fidelity.py`** - Evaluates fidelity metrics
- **`helper.py`** - Utility functions and helper routines
- **`process_experimental.py`** - Processes raw experimental data for inference
- **`simulator.py`** - Simulator of STM spectra
- **`train.py`** - Training routine

### trained_models/ 
- **`model_emax`** - Pretrained model for maximum epsilon
- **`model_emin`** - Pretrained model for minimum epsilon
- **`model_soc`** - Pretrained model for spin-orbit coupling 
- **`model_tau`** - Pretrained model for ferroelectric field
