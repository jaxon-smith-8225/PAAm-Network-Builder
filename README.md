# PAAm-Network-Builder

A coarse-grained Monte Carlo tool for generating crosslinked polyacrylamide (PAAm) hydrogel network topologies and predicting their equilibrium swelling behavior via Flory-Rehner theory.

---

## Overview

Modeling the structure of a chemically crosslinked hydrogel from first principles is non-trivial. Even for a well-characterized system like PAAm/BIS, the relationship between synthesis conditions (monomer concentration, crosslinker ratio) and the resulting network topology — loop density, dangling end fraction, mesh size — is poorly captured by mean-field theories alone. This tool takes a **pack-and-react** approach: polymer chains are represented as cylindrical bounding volumes derived from atomistic PDB templates, stochastically packed into a periodic simulation box at the target volume fraction, and crosslinked according to geometric proximity and radial orientation constraints that mimic the chemistry of BIS-mediated amide coupling.

The resulting network is a graph whose topology can be interrogated directly: cycle rank (the first Betti number), percolation, elastic chain fraction, and mean crosslink spacing are all computed from the bond list. A separate analysis script sweeps crosslink density and runs Flory-Rehner predictions of equilibrium swelling ratio *Q*, bridging the coarse-grained simulation to experimentally measurable quantities.

---

## Physical Model

### Chain Representation

Each PAAm chain is modeled as a **cylinder** defined by its two terminal CTS carbon atoms (axis endpoints) and the Cartesian positions of all pendant amide groups (reactive sites). The cylinder radius is set to 3.8 Å, approximately one monomer width. Reactive sites are located at the carbonyl carbon of each acrylamide unit, identified by scanning the MDAnalysis topology for the pattern C(=O)–NH₂ bonded to a backbone carbon.

Template chains are loaded from PDB files (25- to 50-mer configurations are provided) and used as shape templates: each pack attempt randomly selects a template, applies a uniformly sampled SO(3) rotation, and places the centroid at a random position within the box. Collision detection uses a spatial grid accelerator (DDA traversal, 26-neighbourhood queries) to keep the packing step sub-quadratic.

### Bonding Criterion

Two reactive sites are candidates for a BIS crosslink if:

1. Their Euclidean distance is within the search radius (default: 1.5 × BIS length = 8.1 Å)
2. Their outward radial vectors — from each chain's axis to the site — have a dot product ≤ 0.5 (sites must be roughly facing each other, within ~120°)
3. Neither site is already committed to a bond
4. The two chains do not already share a crosslink (one BIS per chain pair)

Candidates passing these filters are greedily assigned shortest-first to resolve conflicts, ensuring each site appears in at most one bond. This is a reasonable approximation for BIS, which is a short, rigid linker with a fixed end-to-end distance of ~5.4 Å.

### Sol Fraction Removal

After each bonding scan, newly placed cylinders that formed no bonds are removed from the network. This mimics the experimental sol fraction — unreacted chains that wash out during gel purification. All chain IDs and bond indices are remapped after removal to keep the network self-consistent.

### Acceptance Criteria

A network is accepted when it simultaneously satisfies:
- **Crosslink density**: fraction of bonded reactive sites within ±0.15% of the target
- **Percolation**: the largest connected component spans ≥ 60% of chains or ≥ 60% of the box extent in at least one dimension
- **Cycle rank**: number of independent loops ≥ target (default: 4)

Percolation corresponds to the experimental gel point. The cycle rank threshold ensures the network is genuinely multiply connected rather than a branched tree.

### Flory-Rehner Swelling

The equilibrium volumetric swelling ratio *Q* is computed by solving the Flory-Rehner equation numerically (Brent's method):

$$\ln(1 - \phi) + \phi + \chi \phi^2 + V_1 \nu_e \left( \frac{\phi^{1/3}}{\phi_0^{1/3}} - \frac{\phi}{2\phi_0} \right) = 0$$

where φ is the polymer volume fraction at swelling equilibrium, φ₀ is the volume fraction at gelation, χ = 0.47 is the Flory-Huggins parameter for PAAm–water (Gundogan et al., 2004), and νₑ is the effective crosslink density in mol/m³ computed directly from the bond count and box volume. *Q* = 1/φ_eq.

---

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy networkx MDAnalysis matplotlib

# Verify MDAnalysis can read your PDB templates
python -c "import MDAnalysis as mda; u = mda.Universe('PAAm25mer.pdb', context='default', to_guess=['elements','bonds']); print(u.atoms)"
```

Python ≥ 3.9 is recommended. The parallel worker pool uses `ProcessPoolExecutor`; on macOS with Python ≥ 3.12 the `spawn` start method is the default and is handled automatically.

---

## Usage

### Single Network

```bash
# Run with defaults (25 chains, 5% crosslink density, percolation off)
python build_single_network.py

# Match a specific BIS:AAm ratio (e.g. 1:19 ≈ 5.3% of amide sites crosslinked)
python build_single_network.py --crosslink-density 0.053 --volume-fraction 0.08 --chains 30

# Require percolation and a minimum of 6 independent loops
python build_single_network.py --percolation --target-loops 6

# Use specific PDB templates and explicit worker count
python build_single_network.py --pdb PAAm25mer.pdb PAAm35mer.pdb --workers 4
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--crosslink-density` | 0.05 | Target bonded fraction of reactive sites |
| `--volume-fraction` | 0.05 | Polymer volume fraction during gelation |
| `--chains` | 25 | Target chain count |
| `--target-loops` | 4 | Minimum cycle rank for acceptance |
| `--percolation` | off | Require spanning cluster before accepting |
| `--workers` | CPU count | Number of parallel workers |
| `--pdb` | (6 templates) | PDB template files (space-separated) |

### Crosslink Density Sweep

```bash
python full_network_analysis.py
```

This sweeps crosslink densities from 4.5% to 9.5% in 0.2% increments, generates a network at each point, computes topology metrics, solves Flory-Rehner, and saves a 4-panel figure (`network_analysis.png`) showing:
- Swelling ratio *Q* vs. crosslink density
- Mean crosslink spacing ξ (nm)
- Elastic chain fraction (both ends bonded)
- Effective crosslink density νₑ (mol/m³)

The sweep is the primary output for comparison with experimental swelling data.

---

## Configuration

All physical and algorithmic constants live in `config.py`. Key parameters:

```python
CHAIN_RADIUS             = 3.8    # Cylinder radius (Å)
BIS_LENGTH               = 5.4    # BIS end-to-end distance (Å)
BOND_SEARCH_RADIUS       = 8.1    # KD-tree search cutoff (Å)
MIN_RADIAL_DOT_PRODUCT   = 0.5    # Orientation filter threshold
TARGET_CROSSLINK_DENSITY = 0.05   # Default bonded site fraction
POLYMER_VOLUME_FRACTION  = 0.05   # Default φ₀
DEFAULT_CHAIN_COUNT      = 25
BATCH_SIZE               = 8      # Cylinders per pack-and-bond iteration
MAX_ITERATIONS           = 80     # Per-worker iteration limit
REQUIRE_PERCOLATION      = True
TARGET_LOOPS             = 4
```

Tightening `MIN_RADIAL_DOT_PRODUCT` toward −1 enforces stricter antiparallel site orientation; loosening it toward +1 accepts sites regardless of relative orientation. At the default of 0.5, sites within roughly 120 degrees of facing each other are accepted, which is a reasonable approximation for the conformational flexibility of pendant amide groups.

---

## Module Summary

| Module | Responsibility |
|---|---|
| `config.py` | All physical constants and defaults |
| `models.py` | `ChainCylinder` (geometry + transformations) and `CylinderNetwork` (bond graph, topology queries) |
| `geometry_utils.py` | Segment–segment distance, SO(3) rotations, cylinder collision |
| `chain_analysis.py` | MDAnalysis topology scan; amide site detection; cylinder construction from PDB |
| `packing.py` | Spatial grid, SO(3)-uniform random orientation, batch packing |
| `bonding.py` | KD-tree candidate search, orientation filter, greedy conflict resolution |
| `find_structure.py` | Pack-and-bond loop; parallel worker pool (`ProcessPoolExecutor`) |
| `build_single_network.py` | CLI entry point for single network generation |
| `full_network_analysis.py` | Density sweep, Flory-Rehner solver, 4-panel figure |
| `viz.py` | 3D matplotlib visualization; cycle highlighting |

---

## Output and Interpretation

A successful run prints a summary like:

```
Chains:            25
Bonds:             18
Crosslink density: 5.01%
Cycle rank:        5
Percolated:        True
Pack iterations:   14
Bond iterations:   7
```

**Cycle rank** (edges − nodes + connected components) is the physically meaningful loop count — the number of BIS crosslinks in excess of those needed to span the network as a tree. A cycle rank of zero means a branched, tree-like network; higher values indicate a more multiply-connected, elastically efficient gel. This is distinct from the number of simple cycles returned by `find_cycles()`, which grows combinatorially.

**Elastic chain fraction** is computed in `full_network_analysis.py`: a chain is elastically active only if it is bonded at both ends (participates in ≥ 2 distinct crosslinks). Chains bonded at a single end are dangling ends that dissipate rather than store elastic energy under deformation.

---

## Known Limitations

- **Periodic boundary conditions are not enforced.** Chains near the box boundary may extend outside it. Percolation is detected by component size and centroid span, not by explicit image-flag tracking. For small box sizes this can undercount connectivity.
- **Chain flexibility is neglected.** Each chain is a rigid rod between its terminal atoms. Real PAAm is a flexible random coil; the cylindrical model overestimates excluded volume and underestimates the reach of reactive sites near chain ends.
- **One BIS per chain pair.** The `resolve_conflicts` function prevents two bonds between the same pair of chains, which eliminates double-strand crosslinks. This is conservative but avoids graph multigraph complications in the cycle analysis.
- **Box size scales with chain count.** At the default of 25 chains the box is small enough that finite-size effects are non-negligible, particularly for percolation and long-wavelength fluctuations. Increase `--chains` for production runs at the cost of wall-clock time.

---

## References

- Gundogan, N., Melekaslan, D., & Okay, O. (2004). Rubber elasticity of poly(N-isopropylacrylamide) gels at various charge densities. *Macromolecules*, 35(14), 5616–5622. *(χ parameter source)*
- Flory, P. J., & Rehner, J. (1943). Statistical mechanics of cross-linked polymer networks. *Journal of Chemical Physics*, 11(11), 521–526.
- Ericson, C. (2005). *Real-Time Collision Detection*. Morgan Kaufmann. *(segment–segment distance algorithm)*
- Michaud-Agrawal, N., et al. (2011). MDAnalysis: A toolkit for the analysis of molecular dynamics simulations. *Journal of Computational Chemistry*, 32(10), 2319–2327.