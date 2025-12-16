"""
simulator.py

Generates synthetic STM-IETS spectra for the FePc/SnTe system via exact diagonalization.
This module implements the many-body Hamiltonian and computes the spin-flip and orbital
cotunneling contributions to the dynamical correlator S(omega) ~ d^2I/dV^2.

Used for large-scale dataset generation on HPC clusters (e.g., via array jobs).
"""

import numpy as np
import sys
import os
import helper as hp

# --- Path to the DMRGpy library (adjust for your cluster setup) ---
CUSTOM_PATH =  os.environ.get("DMRGPY_PATH")
sys.path.append(os.path.join(CUSTOM_PATH, "src"))

def generate_spectrum(sample_id, num_freq=251, num_eps=80):
    """
    Generates a complete synthetic data sample (spectra + Hamiltonian parameters).

    This function produces the many-body excitation spectrum for a random set of
    Hamiltonian parameters within physically realistic ranges. It computes the
    orbital-cotunneling, spin-flip, and combined orbital-spin-flip channels.

    Parameters
    ----------
    sample_id : int
        Unique identifier for the sample. Used to seed the random number generator,
        ensuring reproducible dataset generation.
    num_freq : int, optional
        Number of energy bias points (default 251).
    num_eps : int, optional
        Number of on-site energy (epsilon) points (default 80).

    Returns
    -------
    dict
        A dictionary containing:
        - 'spectra': complex array of shape (num_eps, num_freq*2)
                     The total dynamical correlator S(omega) for each epsilon.
        - 'spectra_orb': Orbital-cotunneling contribution.
        - 'spectra_flip': Spin-flip contribution.
        - 'spectra_orb_flip': Combined orbital-spin-flip contribution.
        - 'freq_axis': Real energy axis (omega) in eV.
        - 'eps_axis': Tunable d_z² on-site energy axis (epsilon) in eV.
        - 'params': Dictionary of the Hamiltonian parameters used:
                    'tau' (eV), 'lambda_soc' (eV), 'eps_min' (eV), 'eps_max' (eV).
    """

    # Seed for reproducible randomness based on sample ID
    np.random.seed(sample_id)

    # ---------- 1. DEFINE SPECTROSCOPIC AXES ----------
    # Energy bias axis (omega) for the spectrum
    freq_axis = np.linspace(0, 0.20, num_freq)  # eV

    # ---------- 2. SAMPLING OF HAMILTONIAN PARAMETERS ----------
    # Fixed parameters (justified in the Supplementary Information)
    delta_0 = 1.5          # Crystal field splitting (eV)
    theta_deg = 0          # Substrate distortion angle (degrees)
    theta = np.deg2rad(theta_deg)

    # Variable parameters sampled from uniform distributions
    tau = np.random.uniform(0.0, 0.2)      # Ferroelectric field strength (eV)
    lambda_soc = np.random.uniform(0.0, 0.15)  # Spin-orbit coupling (eV)

    # Sampling the tunable d_z² energy window [eps_min, eps_max]
    global_min, global_max = -0.63, 0.0    # Global bounds (eV)
    min_width, max_width = 0.030, 0.3     # Min/Max window width (eV)

    width = np.random.uniform(min_width, max_width)
    center_min = global_min + width / 2
    center_max = global_max - width / 2
    center = np.random.uniform(center_min, center_max)

    eps_min = center - width / 2
    eps_max = center + width / 2

    # Epsilon axis for the 2D spectrum (scaled by factor 10, consistent with hopping scale)
    eps_axis = np.linspace(10 * eps_min, 10 * eps_max, num_eps)

    # ---------- 3. CONSTRUCT THE CRYSTAL-FIELD HAMILTONIAN ----------
    # On-site energies for the 5 Fe 3d orbitals (order: d_z², d_xz, d_yz, d_{x²-y²}, d_xy)
    # The factors of 10 convert from eV to the internal energy scale of the solver.
    onsites = [
        10 * (-0.656 - 1) * delta_0,
        10 * (-1 - 0.656 - 0.086) * delta_0 + 10 * tau * np.cos(2 * theta),
        10 * (-1 - 0.656 - 0.086) * delta_0 - 10 * tau * np.cos(2 * theta),
        0,
        -10 * delta_0
    ]

    tij = np.diag(onsites)
    # Substrate-induced hybridization between d_xz and d_yz
    tij[1, 2] = 10 * tau * np.sin(2 * theta)
    tij[2, 1] = 10 * tau * np.sin(2 * theta)

    # ---------- 4. MANY-BODY SPECTRUM CALCULATION ----------
    # Configuration for the exact diagonalization solver
    solver_mode = "ED"
    solver_submode = "ED"
    broadening = 0.005  # Lorentzian broadening (eV)

    # Initialize arrays for the different spectral channels
    spectra_total = np.zeros((num_eps, 2 * num_freq), dtype=np.complex128)
    spectra_orb = np.zeros_like(spectra_total)
    spectra_flip = np.zeros_like(spectra_total)
    spectra_orb_flip = np.zeros_like(spectra_total)

    # Loop over each value of the tunable on-site energy epsilon
    for eps_idx, eps_value in enumerate(eps_axis):
        # Build the full many-body Hamiltonian (Coulomb U=4.0 eV, Hund's J=0.05 eV)
        hamiltonian, fc = hp.calculate_hamiltonian(
            tij_original=tij,
            epsilon=eps_value,
            mu=0.0,
            U=10 * 4.0,          # Intra-orbital Coulomb (scaled)
            B=0.0,               # Magnetic field
            lambda_soc=lambda_soc * 10,  # Spin-orbit (scaled)
            J=10 * 0.05          # Hund's coupling (scaled)
        )
        fc.set_hamiltonian(hamiltonian)
        fc.kpmmaxm = 20  # Krylov space dimension for the Lanczos solver

        # Compute orbital-cotunneling channel (spin-conserving)
        _, spectra_orb[eps_idx, :] = hp.get_orbital_cotunneling(
            h=hamiltonian, fc=fc, es=freq_axis, num_freq=num_freq,
            mode=solver_mode, submode=solver_submode, delta=broadening
        )

        # Compute spin-flip channel (orbital-conserving)
        _, spectra_flip[eps_idx, :] = hp.get_spinflip(
            h=hamiltonian, fc=fc, es=freq_axis, num_freq=num_freq,
            mode=solver_mode, submode=solver_submode, delta=broadening
        )

        # Compute mixed orbital-spin-flip channel
        _, spectra_orb_flip[eps_idx, :] = hp.get_orbital_spinflip(
            h=hamiltonian, fc=fc, es=freq_axis, num_freq=num_freq,
            mode=solver_mode, submode=solver_submode, delta=broadening
        )

    # Total spectrum is the sum of all contributions
    spectra_total = spectra_orb + spectra_flip + spectra_orb_flip

    # ---------- 5. PACKAGE AND RETURN RESULTS ----------
    results = {
        'spectra': spectra_total,
        'spectra_orb': spectra_orb,
        'spectra_flip': spectra_flip,
        'spectra_orb_flip': spectra_orb_flip,
        'freq_axis': freq_axis,
        'eps_axis': eps_axis / 10.0,  # Convert back to eV for output
        'params': {
            'tau': tau,
            'lambda_soc': lambda_soc,
            'eps_min': eps_min,
            'eps_max': eps_max,
            # Optional: include fixed parameters for completeness
            #'delta_0': delta_0,
            #'theta': theta_deg
        }
    }
    return results

# --- Legacy wrapper for array job compatibility ---
def generate_final_array(sample_id):
    """
    Legacy function for HPC array jobs.
    Generates a flattened array containing all spectra and labels,
    saved as a text file. Compatible with the original dataset generation script.

    Parameters
    ----------
    sample_id : int
        Sample identifier from the array job environment variable.

    Returns
    -------
    np.ndarray
        A 1D array containing concatenated spectra and parameter labels.
    """
    data = generate_spectrum(sample_id)

    # Flatten the complex spectra (real and imaginary parts are interleaved)
    x_flat = data['freq_axis'].astype(np.complex128)  # Frequency axis (reused for shape)
    # This flattening matches the original output format
    spectra_flat = data['spectra'].flatten()
    orb_flat = data['spectra_orb'].flatten()
    flip_flat = data['spectra_flip'].flatten()
    orb_flip_flat = data['spectra_orb_flip'].flatten()

    # Parameter labels
    labels = np.array([
        data['params']['tau'],
        data['params']['lambda_soc'],
        data['params']['eps_min'],
        data['params']['eps_max']
    ])

    # Concatenate everything into a single 1D array
    final_array = np.concatenate([
        x_flat,            # Placeholder for frequency axis (legacy format)
        spectra_flat,
        flip_flat,
        orb_flat,
        orb_flip_flat,
        labels
    ])
    return final_array

if __name__ == "__main__":
    """
    Entry point for HPC array jobs.
    Reads SAMPLE_ID from environment, generates data, and saves to a text file.
    """
    try:
        sample_id = int(os.environ["SAMPLE_ID"])
    except KeyError:
        print("ERROR: Environment variable 'SAMPLE_ID' not set.")
        sys.exit(1)

    final_array = generate_final_array(sample_id)

    output_file = f"final_array_{sample_id:06d}.txt"
    np.savetxt(output_file, final_array, fmt="%.8e")
    print(f"Sample {sample_id} saved to {output_file}")