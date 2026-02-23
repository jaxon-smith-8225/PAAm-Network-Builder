"""
analysis.py — Network topology analysis and Flory-Rehner swelling prediction

Runs the pack-and-bond algorithm across a sweep of crosslink densities,
extracts topologically meaningful metrics from each network, and computes
equilibrium swelling ratios via Flory-Rehner theory.

No modifications to the existing codebase are required.

Usage:
    python analysis.py

Outputs:
    - Console table of metrics per crosslink density
    - network_analysis.png : 4-panel figure
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq

from find_structure import find_structure
from config import (
    BATCH_SIZE,
    MAX_ITERATIONS,
    POLYMER_VOLUME_FRACTION,
    PDB_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
NA      = 6.022e23   # Avogadro's number (mol⁻¹)
V1      = 18.0e-6    # Molar volume of water (m³/mol)
CHI     = 0.47       # Flory-Huggins parameter for PAAm-water (dimensionless)
                     # From Gundogan et al. (2004), Macromol. Chem. Phys.
ANG3_TO_M3 = 1e-30  # Unit conversion: Å³ → m³

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
# Crosslink densities to test (fraction of reactive sites bonded)
TARGET_DENSITIES = np.arange(0.045, 0.095, 0.002)

# Number of chains per network. Keep small for speed on a laptop.
N_CHAINS = 25

# ---------------------------------------------------------------------------
# Topology metrics
# ---------------------------------------------------------------------------

def elastic_chain_fraction(network):
    """
    Fraction of chains that are elastically active.

    A chain contributes to the elastic modulus only if it is bonded at
    both ends (i.e., participates in at least 2 distinct crosslinks).
    Chains bonded at only one end are dangling ends — they relax stress
    and do not store elastic energy.

    Returns:
        float in [0, 1]
    """
    bond_count = {}
    for (a, _, b, _) in network.bonds:
        bond_count[a] = bond_count.get(a, 0) + 1
        bond_count[b] = bond_count.get(b, 0) + 1

    n_active = sum(
        1 for cid in range(network.num_chains())
        if bond_count.get(cid, 0) >= 2
    )
    return n_active / max(network.num_chains(), 1)


def effective_crosslink_density_mol_m3(network):
    """
    Compute ν_e: moles of crosslinks per cubic metre of network.

    Each entry in network.bonds is one BIS crosslinker connecting two
    chain sites. Dividing by Avogadro's number and box volume gives a
    physical crosslink density in SI units suitable for Flory-Rehner.

    Returns:
        float (mol/m³)
    """
    n_bonds       = len(network.bonds)
    box_vol_m3    = float(np.prod(network.box_dims)) * ANG3_TO_M3
    return n_bonds / (NA * box_vol_m3)


def mean_crosslink_spacing_nm(network):
    """
    Mean Euclidean distance between bonded reactive site pairs (in nm).

    This is a geometric proxy for the network mesh size ξ. The true mesh
    size depends on chain stiffness and contour length, but this spatial
    measure captures how crosslink density compresses the mesh.

    Returns:
        float (nm), or np.nan if no bonds present
    """
    if not network.bonds:
        return np.nan

    distances = []
    for (a, sa, b, sb) in network.bonds:
        pa = network.cylinders[a].linking_points[sa]
        pb = network.cylinders[b].linking_points[sb]
        distances.append(np.linalg.norm(pa - pb))

    return float(np.mean(distances)) / 10.0   # Å → nm


# ---------------------------------------------------------------------------
# Flory-Rehner theory
# ---------------------------------------------------------------------------

def flory_rehner_swelling_ratio(nu_e, phi_0=POLYMER_VOLUME_FRACTION, chi=CHI, v1=V1):
    """
    Solve the Flory-Rehner equation for equilibrium volumetric swelling ratio Q.

    The equation balances the osmotic driving force for swelling (mixing term)
    against the elastic restoring force of the crosslinked network:

        ln(1 - φ) + φ + χ·φ² + V₁·νₑ·(φ^(1/3)/φ₀^(1/3) - φ/(2·φ₀)) = 0

    where φ is the polymer volume fraction at swelling equilibrium and φ₀ is
    the polymer volume fraction at the time of network formation (our
    POLYMER_VOLUME_FRACTION). The equation is solved numerically via bisection.

    Q = 1/φ_eq  →  Q > 1 means the gel takes up water (swells).

    Args:
        nu_e  : Effective crosslink density (mol/m³)
        phi_0 : Polymer volume fraction during gelation
        chi   : Flory-Huggins solvent-polymer interaction parameter
        v1    : Molar volume of solvent (m³/mol)

    Returns:
        float Q (dimensionless), or np.nan if the solver fails
    """
    phi0_cbrt = phi_0 ** (1.0 / 3.0)

    def equation(phi):
        mixing  = np.log(1.0 - phi) + phi + chi * phi ** 2
        elastic = v1 * nu_e * (phi ** (1.0 / 3.0) / phi0_cbrt - phi / (2.0 * phi_0))
        return mixing + elastic

    # The solution must lie between a tiny value and just below 1
    try:
        # Confirm sign change exists before calling brentq
        f_lo = equation(1e-6)
        f_hi = equation(1.0 - 1e-6)
        if f_lo * f_hi > 0:
            return np.nan
        phi_eq = brentq(equation, 1e-6, 1.0 - 1e-6, xtol=1e-10)
        return 1.0 / phi_eq
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_analysis(target_densities=TARGET_DENSITIES, n_chains=N_CHAINS):
    """
    Run the pack-and-bond algorithm at each crosslink density and collect metrics.

    Returns:
        List of result dicts, one per density target.
    """
    results = []

    print("=" * 62)
    print(f"{'Density':>10} {'Q':>8} {'Mesh(nm)':>10} {'ν_e(mol/m³)':>13} {'ElasticFrac':>12} {'CycleRank':>10}")
    print("-" * 62)

    for density in target_densities:
        cfg = {
            'pdb_template':             PDB_TEMPLATE,
            'target_chains':            n_chains,
            'batch_size':               BATCH_SIZE,
            'max_iterations':           MAX_ITERATIONS,
            'target_crosslink_density': density,
            'polymer_volume_fraction':  POLYMER_VOLUME_FRACTION,
            'require_percolation':      True,
            'target_loops':             3,    # relax loop requirement for low densities
        }

        network, stats = find_structure(cfg)

        if network is None:
            print(f"{density*100:>9.1f}%  {'FAILED':>8}")
            results.append({
                'target_density':   density,
                'actual_density':   np.nan,
                'Q':                np.nan,
                'mesh_nm':          np.nan,
                'nu_e':             np.nan,
                'elastic_fraction': np.nan,
                'cycle_rank':       np.nan,
            })
            continue

        nu_e  = effective_crosslink_density_mol_m3(network)
        Q     = flory_rehner_swelling_ratio(nu_e)
        mesh  = mean_crosslink_spacing_nm(network)
        ef    = elastic_chain_fraction(network)
        rank  = stats['cycle_rank']
        actual_density = stats['crosslink_density']

        print(
            f"{actual_density*100:>9.1f}%"
            f"  {Q:>8.1f}"
            f"  {mesh:>10.2f}"
            f"  {nu_e:>13.2f}"
            f"  {ef*100:>10.1f}%"
            f"  {rank:>10}"
        )

        results.append({
            'target_density':   density,
            'actual_density':   actual_density,
            'Q':                Q,
            'mesh_nm':          mesh,
            'nu_e':             nu_e,
            'elastic_fraction': ef,
            'cycle_rank':       rank,
        })

    print("=" * 62)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, save_path='network_analysis.png'):
    """
    Produce a 4-panel figure summarising network properties across crosslink densities.
    """
    # Filter out failed runs
    valid = [r for r in results if not np.isnan(r['Q'])]
    if not valid:
        print("No valid results to plot.")
        return

    densities_pct    = [r['actual_density'] * 100 for r in valid]
    Q_vals           = [r['Q']                    for r in valid]
    mesh_vals        = [r['mesh_nm']               for r in valid]
    nu_e_vals        = [r['nu_e']                  for r in valid]
    ef_vals          = [r['elastic_fraction'] * 100 for r in valid]

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle('PAAm Hydrogel Network Analysis (Flory-Rehner Theory)', fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # --- Panel 1: Swelling ratio vs crosslink density ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(densities_pct, Q_vals, 'o-', color='steelblue', linewidth=2,
             markersize=7, label='Flory-Rehner prediction')

    ax1.set_xlabel('Crosslink Density (%)')
    ax1.set_ylabel('Swelling Ratio  Q  (V_swollen / V_dry)')
    ax1.set_title('Equilibrium Swelling Ratio')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Mesh size ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(densities_pct, mesh_vals, 's-', color='darkorange', linewidth=2, markersize=7)
    ax2.set_xlabel('Crosslink Density (%)')
    ax2.set_ylabel('Mean Crosslink Spacing  ξ  (nm)')
    ax2.set_title('Network Mesh Size')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Elastic chain fraction ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(densities_pct, ef_vals, '^-', color='seagreen', linewidth=2, markersize=7)
    ax3.set_xlabel('Crosslink Density (%)')
    ax3.set_ylabel('Elastically Active Chains (%)')
    ax3.set_title('Elastic Chain Fraction\n(both ends bonded)')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Effective crosslink density ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(densities_pct, nu_e_vals, 'D-', color='mediumpurple', linewidth=2, markersize=7)
    ax4.set_xlabel('Crosslink Density (%)')
    ax4.set_ylabel('ν_e  (mol / m³)')
    ax4.set_title('Effective Crosslink Density\n(physical units)')
    ax4.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    results = run_analysis()
    plot_results(results)