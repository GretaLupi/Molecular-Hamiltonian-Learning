import numpy as np
import pandas as pd
from dmrgpy import fermionchain  # DMRG library for many-body calculations
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

def calculate_hamiltonian(tij_original, epsilon=0., U=0., B=0., lambda_soc=0., J=0.):
    """
    Constructs the many-body Hamiltonian for a 5-orbital system (e.g., Fe2+ d-orbitals)
    with optional single-particle terms, electron-electron interactions, and spin-orbit coupling.

    Args:
        tij_original (np.ndarray): 5x5 matrix of bare hopping integrals/orbital energies.
        epsilon (float): Local energy shift applied to the first orbital (e.g., d_z^2).
        U (float): Strength of the spin-spin (S^2) Coulomb interaction term (-U * S^2).
        B (float): Strength of the external Zeeman magnetic field along the z-axis.
        lambda_soc (float): Strength of the atomic spin-orbit coupling.
        J (float): Strength of the orbital-orbital (L^2) Coulomb interaction term (-J * L^2).

    Returns:
        tuple: (h, fc)
            h: The constructed Hamiltonian as a fermionic operator.
            fc: The fermionchain object, with added attributes for total spin (ST) and orbital (LT) operators.
    """

    # --------------------------------------------------------------------
    # 1. TRANSFORMATION MATRICES: Real Spherical Harmonics <-> Complex d-orbitals
    # --------------------------------------------------------------------
    # Transformation matrix T converts from the complex spherical harmonic basis
    # (|2>, |1>, |0>, |-1>, |-2>) to the real cubic harmonic basis (d_z^2, d_xz, d_yz, d_x^2-y^2, d_xy).
    # Its conjugate transpose (Td) performs the reverse transformation.
    T = np.array([
        [0, 0, 0, 1/np.sqrt(2), -1j/np.sqrt(2)],  
        [0, -1/np.sqrt(2), +1j/np.sqrt(2), 0, 0],  
        [1, 0, 0, 0, 0],                           
        [0, 1/np.sqrt(2), +1j/np.sqrt(2), 0, 0],   
        [0, 0, 0, 1/np.sqrt(2), +1j/np.sqrt(2)]    
    ])
    Td = np.conjugate(T.T) 

    # --------------------------------------------------------------------
    # 2. ANGULAR MOMENTUM OPERATORS in the Complex Spherical Harmonic Basis (L=2)
    # --------------------------------------------------------------------
    hbar = 1  # Setting hbar = 1 for simplicity.
    # Lz is diagonal in this basis with eigenvalues m_l = 2, 1, 0, -1, -2.
    Lzp = hbar * np.diag([2, 1, 0, -1, -2])

    # Matrix elements for L+ (raising operator) in the |l, m> basis.
    # Lx = (L+ + L-)/2, Ly = (L+ - L-)/(2i)
    Lxp = (hbar / 2) * np.array([
        [0, 2, 0, 0, 0],
        [2, 0, np.sqrt(6), 0, 0],
        [0, np.sqrt(6), 0, np.sqrt(6), 0],
        [0, 0, np.sqrt(6), 0, 2],
        [0, 0, 0, 2, 0]
    ])
    Lyp = (hbar / (2j)) * np.array([
        [0, 2, 0, 0, 0],
        [-2, 0, np.sqrt(6), 0, 0],
        [0, -np.sqrt(6), 0, np.sqrt(6), 0],
        [0, 0, -np.sqrt(6), 0, 2],
        [0, 0, 0, -2, 0]
    ])

    # --------------------------------------------------------------------
    # 3. TRANSFORM ANGULAR MOMENTUM OPERATORS
    # --------------------------------------------------------------------
    lz = Td @ Lzp @ T
    lx = Td @ Lxp @ T
    ly = Td @ Lyp @ T

    # --------------------------------------------------------------------
    # 4. SPIN-1/2 OPERATORS
    # --------------------------------------------------------------------
    sigma_z = 0.5 * np.array([[1, 0], [0, -1]])
    sigma_x = 0.5 * np.array([[0, 1], [1, 0]])
    sigma_y = 0.5 * np.array([[0, -1j], [1j, 0]])

    #################################################################
    # 5. CONSTRUCT THE FERMIONIC CHAIN AND HAMILTONIAN
    #################################################################
    n = 5  # Number of d orbitals
    fc = fermionchain.Spinful_Fermionic_Chain(n)  # Create chain with spinful fermions
    h = 0  # Initialize Hamiltonian operator

    # --------------------------------------------------------------------
    # 5a. SINGLE-PARTICLE (NON-INTERACTING) TERMS
    # --------------------------------------------------------------------
    tij = np.copy(tij_original)
    # Apply the local energy shift 'epsilon' to the first orbital (index 0).
    # This models, e.g., a crystal field shift of the d_z^2 orbital.
    tij[0][0] = tij_original[0][0] + epsilon

    # Add all spin-independent hopping terms.
    # fc.Cdagup[i] * fc.Cup[j] annihilates spin-up at site j and creates at i.
    for i in range(n):
        for j in range(n):
            h += tij[i, j] * (fc.Cdagup[i] * fc.Cup[j] + fc.Cdagdn[i] * fc.Cdn[j])

    # --------------------------------------------------------------------
    # 5b. PARTICLE NUMBER CONSTRAINT (LAGRANGE MULTIPLIER)
    # --------------------------------------------------------------------
    # Enforces exactly 6 electrons (for Fe2+ configuration) with a large penalty.
    N_tot = sum(fc.N)  # Total particle number operator
    h += 100 * (N_tot - 6) * (N_tot - 6)

    h = h + h.get_dagger()

    # --------------------------------------------------------------------
    # 5c. SPIN-ORBIT COUPLING (SOC) TERM: lambda_soc * (L · S)
    # --------------------------------------------------------------------
    for i in range(n):
        for j in range(n):
            for s in range(2):  
                for sp in range(2):  
                    for L_op, sigma_op in [(lz, sigma_z), (lx, sigma_x), (ly, sigma_y)]:
                        coeff = lambda_soc * L_op[i, j] * sigma_op[s, sp]
                        if s == 0 and sp == 0:
                            op = coeff * fc.Cdagup[i] * fc.Cup[j]
                        elif s == 1 and sp == 1:
                            op = coeff * fc.Cdagdn[i] * fc.Cdn[j]
                        elif s == 0 and sp == 1:
                            op = coeff * fc.Cdagup[i] * fc.Cdn[j]
                        elif s == 1 and sp == 0:
                            op = coeff * fc.Cdagdn[i] * fc.Cup[j]
                        else:
                            op = 0
                        h += op

    # --------------------------------------------------------------------
    # 5d. CONSTRUCT TOTAL SPIN (S) AND ORBITAL (L) ANGULAR MOMENTUM OPERATORS
    # --------------------------------------------------------------------
    Lz = Lx = Ly = Sz = Sx = Sy = 0
    # Build L components from single-particle orbital operators.
    for i in range(n):
        for j in range(n):
            Lz += lz[i, j] * (fc.Cdagup[i] * fc.Cup[j] + fc.Cdagdn[i] * fc.Cdn[j])
            Lx += lx[i, j] * (fc.Cdagup[i] * fc.Cup[j] + fc.Cdagdn[i] * fc.Cdn[j])
            Ly += ly[i, j] * (fc.Cdagup[i] * fc.Cup[j] + fc.Cdagdn[i] * fc.Cdn[j])
        # Build S components from on-site spin density.
        for s in range(2):
            for sp in range(2):
                Sz += sigma_z[s, sp] * (
                    fc.Cdagup[i] * fc.Cup[i] if s == 0 and sp == 0 else
                    fc.Cdagdn[i] * fc.Cdn[i] if s == 1 and sp == 1 else
                    fc.Cdagup[i] * fc.Cdn[i] if s == 0 and sp == 1 else
                    fc.Cdagdn[i] * fc.Cup[i] if s == 1 and sp == 0 else 0
                )
                Sx += sigma_x[s, sp] * (
                    fc.Cdagup[i] * fc.Cup[i] if s == 0 and sp == 0 else
                    fc.Cdagdn[i] * fc.Cdn[i] if s == 1 and sp == 1 else
                    fc.Cdagup[i] * fc.Cdn[i] if s == 0 and sp == 1 else
                    fc.Cdagdn[i] * fc.Cup[i] if s == 1 and sp == 0 else 0
                )
                Sy += sigma_y[s, sp] * (
                    fc.Cdagup[i] * fc.Cup[i] if s == 0 and sp == 0 else
                    fc.Cdagdn[i] * fc.Cdn[i] if s == 1 and sp == 1 else
                    fc.Cdagup[i] * fc.Cdn[i] if s == 0 and sp == 1 else
                    fc.Cdagdn[i] * fc.Cup[i] if s == 1 and sp == 0 else 0
                )

    # Square of total S and L operators: S^2 = Sx^2 + Sy^2 + Sz^2, similarly for L^2.
    S2 = Sz * Sz + Sx * Sx + Sy * Sy
    L2 = Lz * Lz + Lx * Lx + Ly * Ly

    # --------------------------------------------------------------------
    # 5e. ADD ELECTRON-ELECTRON INTERACTION TERMS
    # --------------------------------------------------------------------
    # -U * S^2 : Favors high-spin states (Hund's rule coupling).
    # -J * L^2 : Favors high orbital angular momentum states.
    h -= U * S2 + J * L2

    # --------------------------------------------------------------------
    # 5f. EXTERNAL MAGNETIC FIELD (ZEEEMAN TERM)
    # --------------------------------------------------------------------
    # B * (N_up - N_down) = B * S_z (up to factor of 2, depending on definition of sigma_z).
    for i in range(n):
        h += B * (fc.Nup[i] - fc.Ndn[i])

    # --------------------------------------------------------------------
    # 5g. ATTACH TOTAL OPERATORS TO THE CHAIN OBJECT FOR LATER USE
    # --------------------------------------------------------------------
    fc.STx = Sx
    fc.STy = Sy
    fc.STz = Sz
    fc.LTx = Lx
    fc.LTy = Ly
    fc.LTz = Lz

    return h, fc


def get_orbital_cotunneling(h, fc, es, num_freq, mode, submode, delta):
    """
    Calculates the dynamical correlator for spin-conserving orbital cotunneling.
    Specifically, computes the spectral function for processes where an electron
    tunnels between the central orbital (index 0) and a target orbital (1-4) WITHOUT flipping spin.

    This models inelastic tunneling features in dI/dV where the molecular state changes
    due to orbital excitations, but the total spin projection remains the same.

    Args:
        h: Hamiltonian (not directly used, but fc is derived from it).
        fc: Fermion chain object with the Hamiltonian set.
        es (np.ndarray): Energy grid for the dynamical correlator output.
        num_freq (int): Number of frequency points (length of es).
        mode (str): DMRG calculation mode (e.g., 'ED' for exact diagonalization).
        submode (str): Further specification for the DMRG engine.
        delta (float): Broadening factor (imaginary part added to energy) for the spectral function.

    Returns:
        tuple: (x_combined, y_combined)
            x_combined: Combined frequency array (negative and positive energies).
            y_combined: Combined spectral function (cotunneling intensity).
    """
    # Initialize arrays for spin-up and spin-down spectral functions.
    y_up_pos = np.zeros(num_freq, dtype=np.complex128)
    y_dn_pos = np.zeros(num_freq, dtype=np.complex128)
    y_up_neg = np.zeros(num_freq, dtype=np.complex128)
    y_dn_neg = np.zeros(num_freq, dtype=np.complex128)

    wf = fc.get_gs(mode=mode)  # Get the many-body ground state wavefunction.
    total_y_pos = np.zeros(num_freq, dtype=np.complex128)
    total_y_neg = np.zeros(num_freq, dtype=np.complex128)

    # Loop over all target orbitals (indices 1 to 4).
    for target_orbital in range(1, 5):
        # ------- SPIN-UP CHANNEL -------
        # Operator for 0 -> target (spin-up): c_{0,up}^dagger c_{target,up}
        A01_up = fc.Cdagup[0] * fc.Cup[target_orbital]
        A01_up = A01_up - wf.dot(A01_up * wf)
        T01_up = (A01_up, A01_up.get_dagger())  # Correlator for the (0->target) process
        A10_up = fc.Cdagup[target_orbital] * fc.Cup[0]
        A10_up = A10_up - wf.dot(A10_up * wf)
        T10_up = (A10_up, A10_up.get_dagger())  # Correlator for the (target->0) process

        # Get dynamical correlators for spin-up.
        x_pos, y_up_pos = fc.get_dynamical_correlator(name=T01_up, mode=mode, submode=submode, es=es, delta=delta)
        x_neg, y_up_neg = fc.get_dynamical_correlator(name=T10_up, mode=mode, submode=submode, es=es, delta=delta)

        # ------- SPIN-DOWN CHANNEL (identical structure) -------
        A01_dn = fc.Cdagdn[0] * fc.Cdn[target_orbital]
        A01_dn = A01_dn - wf.dot(A01_dn * wf)
        T01_dn = (A01_dn, A01_dn.get_dagger())

        A10_dn = fc.Cdagdn[target_orbital] * fc.Cdn[0]
        A10_dn = A10_dn - wf.dot(A10_dn * wf)
        T10_dn = (A10_dn, A10_dn.get_dagger())

        _, y_dn_pos = fc.get_dynamical_correlator(name=T01_dn, mode=mode, submode=submode, es=es, delta=delta)
        _, y_dn_neg = fc.get_dynamical_correlator(name=T10_dn, mode=mode, submode=submode, es=es, delta=delta)

        x_neg = -x_neg

        y_neg = y_up_neg + y_dn_neg
        y_pos = y_up_pos + y_dn_pos

        # Sort the negative energy results for clean concatenation.
        sorted_indices_neg = np.argsort(x_neg)
        x_neg = x_neg[sorted_indices_neg]
        y_neg = y_neg[sorted_indices_neg]

        # Accumulate contributions from different orbital pairs.
        total_y_neg += y_neg
        total_y_pos += y_pos

    x_combined = np.concatenate([x_neg, x_pos])
    y_combined = np.concatenate([total_y_neg, total_y_pos])

    return x_combined, y_combined


def get_orbital_spinflip(h, fc, es, num_freq, mode, submode, delta):
    """
    Calculates the dynamical correlator for orbital-assisted spin-flip tunneling.
    Computes the spectral function for processes where an electron tunnels between
    the central orbital (0) and a target orbital (1-4) WITH a flip of its spin.

    This models inelastic tunneling features where both the orbital and the total
    spin state of the molecule change (e.g., spin excitations coupled to orbital degrees of freedom).

    Args and Returns: Same structure as `get_orbital_cotunneling`.
    """

    y_up_pos = np.zeros(num_freq, dtype=np.complex128)
    y_dn_pos = np.zeros(num_freq, dtype=np.complex128)
    y_up_neg = np.zeros(num_freq, dtype=np.complex128)
    y_dn_neg = np.zeros(num_freq, dtype=np.complex128)

    wf = fc.get_gs(mode=mode)
    total_y_pos = np.zeros(num_freq, dtype=np.complex128)
    total_y_neg = np.zeros(num_freq, dtype=np.complex128)

    for target_orbital in range(1, 5):
        # ------- Process: Spin UP on orbital 0 -> Spin DOWN on target orbital -------
        # Operator: c_{0,up}^dagger c_{target,down}
        A01_up_dn = fc.Cdagup[0] * fc.Cdn[target_orbital]
        A01_up_dn = A01_up_dn - wf.dot(A01_up_dn * wf)
        T01_up_dn = (A01_up_dn, A01_up_dn.get_dagger())

        # Reverse process: c_{target,down}^dagger c_{0,up}
        A10_up_dn = fc.Cdagdn[target_orbital] * fc.Cup[0]
        A10_up_dn = A10_up_dn - wf.dot(A10_up_dn * wf)
        T10_up_dn = (A10_up_dn, A10_up_dn.get_dagger())

        x_pos, y_up_pos = fc.get_dynamical_correlator(name=T01_up_dn, mode=mode, submode=submode, es=es, delta=delta)
        x_neg, y_up_neg = fc.get_dynamical_correlator(name=T10_up_dn, mode=mode, submode=submode, es=es, delta=delta)

        # ------- Process: Spin DOWN on orbital 0 -> Spin UP on target orbital -------
        A01_dn_up = fc.Cdagdn[0] * fc.Cup[target_orbital]
        A01_dn_up = A01_dn_up - wf.dot(A01_dn_up * wf)
        T01_dn_up = (A01_dn_up, A01_dn_up.get_dagger())

        A10_dn_up = fc.Cdagup[target_orbital] * fc.Cdn[0]
        A10_dn_up = A10_dn_up - wf.dot(A10_dn_up * wf)
        T10_dn_up = (A10_dn_up, A10_dn_up.get_dagger())

        _, y_dn_pos = fc.get_dynamical_correlator(name=T01_dn_up, mode=mode, submode=submode, es=es, delta=delta)
        _, y_dn_neg = fc.get_dynamical_correlator(name=T10_dn_up, mode=mode, submode=submode, es=es, delta=delta)

        x_neg = -x_neg

        y_neg = y_up_neg + y_dn_neg
        y_pos = y_up_pos + y_dn_pos

        sorted_indices_neg = np.argsort(x_neg)
        x_neg = x_neg[sorted_indices_neg]
        y_neg = y_neg[sorted_indices_neg]

        total_y_neg += y_neg
        total_y_pos += y_pos

    x_combined = np.concatenate([x_neg, x_pos])
    y_combined = np.concatenate([total_y_neg, total_y_pos])

    return x_combined, y_combined


def get_spinflip(h, fc, es, num_freq, mode, submode, delta):
    """
    Calculates the dynamical spin susceptibility (spin-spin correlator).
    Specifically, computes << S^alpha; S^alpha >> for alpha = x, y, z and sums them,
    which gives the spectrum of spin-flip excitations (e.g., magnons, spin waves).

    This is proportional to the inelastic tunneling signal in dI/dV when the tunneling
    electron exchanges spin with the system (e.g., via exchange scattering).

    Args:
        h: Hamiltonian (not directly used).
        fc: Fermion chain object with site spin operators fc.Sx[i], etc.
        es, num_freq, mode, submode, delta: As in the previous functions.

    Returns:
        tuple: (x_array, y_accum)
            x_array: Combined frequency array (negative and positive energies).
            y_accum: Total dynamical spin structure factor (sum over x, y, z directions).
    """
    y_accum = np.zeros(num_freq * 2, dtype=np.complex128)

    # Define spin fluctuation operators: S_i^alpha - <S_i^alpha>_gs
    Sx_fluct = fc.Sx[0] - fc.vev(fc.Sx[0], mode=mode)
    Sy_fluct = fc.Sy[0] - fc.vev(fc.Sy[0], mode=mode)
    Sz_fluct = fc.Sz[0] - fc.vev(fc.Sz[0], mode=mode)

    # Correlators for each spin component: << S^alpha; S^alpha >>
    name_x = (Sx_fluct, Sx_fluct)
    name_y = (Sy_fluct, Sy_fluct)
    name_z = (Sz_fluct, Sz_fluct)

    _, y_x = fc.get_dynamical_correlator(delta=delta, name=name_x, es=es, mode=mode, submode=submode)
    _, y_y = fc.get_dynamical_correlator(delta=delta, name=name_y, es=es, mode=mode, submode=submode)
    x, y_z = fc.get_dynamical_correlator(delta=delta, name=name_z, es=es, mode=mode, submode=submode)

    # Sum contributions from all spin directions.
    y_sum = y_x + y_y + y_z

    y_neg = y_sum
    x_neg = -x

    sorted_indices_neg = np.argsort(x_neg)
    x_neg = x_neg[sorted_indices_neg]
    y_neg = y_neg[sorted_indices_neg]

    y_accum = np.concatenate([y_neg, y_sum])
    x_array = np.concatenate([x_neg, x])

    return x_array, y_accum

def compute_integral_array(x_array, y_array, num_freq, num_eps):
    """
    Computes the cumulative integral of y with respect to ω for each ε and sample.
    This is used to transform spectral functions (dI/dV) into integrated spectral weight,
    which often better reveals step-like features in inelastic tunneling spectroscopy.

    Args:
        x_array: np.ndarray shape (n_samples, 2*num_freq*num_eps)
                 Array of frequency values ω. Each sample contains num_eps blocks of size 2*num_freq.
                 Each block contains: [negative frequencies (sorted descending), positive frequencies (sorted ascending)].
        y_array: np.ndarray shape (n_samples, 2*num_freq*num_eps)
                 Array of spectral function values A(ω,ε) corresponding to x_array.
        num_freq: Number of positive frequency points per block (half of block_size).
        num_eps: Number of ε (e.g., tip-height or gate voltage) values (blocks per sample).

    Returns:
        integral_array: np.ndarray shape (n_samples, 2*num_freq*num_eps)
                       Same shape as input, but with y integrated cumulatively along ω.
                       For each (sample, ε) block: ∫_{-∞}^{ω} A(ω',ε) dω'.
    """
    n_samples = x_array.shape[0]
    block_size = 2 * num_freq  # Each ε-block contains both negative and positive frequencies
    integral_array = np.zeros_like(y_array)
    
    # Loop over all samples (e.g., different Hamiltonian parameters)
    for i in range(n_samples):
        # Loop over each ε value (different tip heights/gate voltages)
        for eps_idx in range(num_eps):
            start = eps_idx * block_size
            end = (eps_idx + 1) * block_size
            
            # Extract frequency and spectral function for this (sample, ε) block
            x = x_array[i, start:end].real  # Frequency ω (real part)
            y = y_array[i, start:end].real  # Spectral function A(ω,ε) (real part)
            
            mid = block_size // 2  # Index separating negative and positive frequencies
            
            # ======================= POSITIVE FREQUENCIES =======================
            x_pos = x[mid:]
            y_pos = y[mid:]
            sort_idx_pos = np.argsort(x_pos)
            x_pos_sorted = x_pos[sort_idx_pos]
            y_pos_sorted = y_pos[sort_idx_pos]
            
            y_int_pos = cumulative_trapezoid(y_pos_sorted, x_pos_sorted, initial=0)
            
            y_int_pos_unsorted = np.empty_like(y_int_pos)
            y_int_pos_unsorted[sort_idx_pos] = y_int_pos
            
            # ======================= NEGATIVE FREQUENCIES =======================
            x_neg = x[:mid] 
            y_neg = y[:mid]  
            
            sort_idx_neg = np.argsort(-x_neg) 
            x_neg_sorted = x_neg[sort_idx_neg]
            y_neg_sorted = y_neg[sort_idx_neg]
            
            y_int_neg = cumulative_trapezoid(y_neg_sorted, x_neg_sorted, initial=0)
            y_int_neg = np.abs(y_int_neg)
            
            inverse_sort_idx_neg = np.argsort(sort_idx_neg)
            y_int_neg_unsorted = np.empty_like(y_int_neg)
            y_int_neg_unsorted[inverse_sort_idx_neg] = y_int_neg
            
            # ======================= COMBINE NEGATIVE AND POSITIVE =======================
            y_integrated = np.zeros_like(y)
            y_integrated[:mid] = y_int_neg_unsorted    # Negative ω region
            y_integrated[mid:] = y_int_pos_unsorted    # Positive ω region
            
            # Store in output array
            integral_array[i, start:end] = y_integrated
    
    return integral_array

#### UTILITIES FOR PREPROCESSING AND VALIDATION #####

def fidelity(true_values, predicted_values):
    """Compute fidelity (correlation) between true and predicted values."""
    num = np.mean(predicted_values * true_values) - np.mean(predicted_values) * np.mean(true_values)
    den = np.sqrt(np.var(true_values)) * np.sqrt(np.var(predicted_values))
    return num / (den + 1e-12)

def per_sample_normalization(X, mode='max'):
    """Normalize each sample independently."""
    X = X.astype(np.float32, copy=False)
    if mode == 'max':
        s = np.max(np.abs(X), axis=-1, keepdims=True)
    else:  # RMS
        s = np.sqrt(np.mean(X**2, axis=-1, keepdims=True))
    return X / (s + 1e-12)

def plot_training_history(history, model_name, output_dir):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training History - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/plots/loss_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_parity_plot(y_true, y_pred, model_name, fidelity, output_dir):
    """Create parity plot for model predictions."""
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
    plt.title(f'{model_name} | Fidelity = {fidelity:.4f}')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/parity_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def make_keep_mask_for_omega(omega, atol=1e-15):
    """
    Create mask to remove duplicate zero point in frequency array.
    
    In the raw DMRG output, there's often a duplicate ω=0 point 
    (one for positive side, one for negative side). This function
    identifies and removes one of them to avoid redundancy.
    
    Args:
        omega: Frequency array (1D)
        atol: Absolute tolerance for identifying zero
    
    Returns:
        keep_mask: Boolean mask (True = keep, False = drop)
    """
    omega = np.asarray(omega, dtype=float)
    
    # Find indices close to zero
    zero_indices = np.where(np.isclose(omega, 0.0, atol=atol))[0]
    
    if len(zero_indices) == 0:
        raise ValueError("No zero point found in omega array.")
    
    # Find the zero point on the negative side (if exists)
    negative_zeros = zero_indices[omega[zero_indices] < 0]
    
    if len(negative_zeros) > 0:
        # Remove the negative-side zero
        drop_idx = negative_zeros[0]
    else:
        # Fallback: remove the point just before middle
        drop_idx = len(omega) // 2 - 1
    
    # Create mask (keep all except the duplicate zero)
    keep_mask = np.ones(len(omega), dtype=bool)
    keep_mask[drop_idx] = False
    
    return keep_mask


def drop_omega_idx_in_flat(X_flat, num_eps, keep_mask):
    """
    Remove duplicate zero frequency point from flattened spectra.
    
    Args:
        X_flat: Flattened spectra shape (n_samples, num_eps * (2*num_freq))
        num_eps: Number of epsilon values
        keep_mask: Boolean mask from make_keep_mask_for_omega
    
    Returns:
        X_dropped: Spectra with duplicate zero removed
    """
    # Repeat the mask for each epsilon curve
    keep_full = np.tile(keep_mask, num_eps)
    
    return X_flat[:, keep_full]

def add_noise(data, std_dev=0.01):
    """Add Gaussian noise to data."""
    return data + np.random.normal(0, std_dev, size=data.shape)