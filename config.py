"""Configuration constants for polymer network simulation"""

import numpy as np

# ---------------------------------------------------------------------------
# Geometric parameters
# ---------------------------------------------------------------------------
CHAIN_RADIUS = 3.8   # Angstroms — radius of cylindrical bounding volume
BIS_LENGTH = 5.4     # Angstroms — length of BIS crosslinker

# ---------------------------------------------------------------------------
# Pack-and-react parameters
# ---------------------------------------------------------------------------

# Fraction of reactive sites that must bond before a network is accepted
# Maps directly to the experimental BIS:acrylamide molar ratio.
TARGET_CROSSLINK_DENSITY = 0.05

# Volume fraction of polymer in the gel-forming solution
# 8% w/v PAAm ≈ 0.06 volume fraction.
# Set lower to give more packing room in the simulation box when chain count is small
POLYMER_VOLUME_FRACTION = 0.05

# Cylinders to attempt placing per pack-and-react iteration
# Larger batches converge faster but may waste effort if the box is dense
BATCH_SIZE = 8

# Max number of pack-and-react iterations per worker before giving up
MAX_ITERATIONS = 80

# Whether the network must percolate (span the box) before being accepted
# Percolation corresponds to the experimental gel point --> probably want it to be True
# Disabled by default — it is the hardest criterion to satisfy with small chain counts and short BIS
#   Enable with --percolation on the command line
REQUIRE_PERCOLATION = True

# Minimum cycle rank (number of independent loops) for acceptance
# TARGET_LOOPS = 0 to disable loop count requirement
TARGET_LOOPS = 4

# Search radius used to find bond candidates in the KD-tree scan
BOND_SEARCH_RADIUS = BIS_LENGTH * 1.5   # 8.1 Å

# Dot product threshold for radial orientation compatibility
# Reactive sites can bond if the dot product of their outward radial vectors
# is <= this value. 0.5 is permissive (sites within ~120 degrees of facing
# each other are accepted)
# Closer to -0.5, the tighter the constraint
MIN_RADIAL_DOT_PRODUCT = 0.5

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DEFAULT_CHAIN_COUNT = 25   # Target number of chains in the final network

# PDB template for the PAAm chain
# PDB_TEMPLATE = ["PAAm25mer.pdb", "PAAm30mer.pdb", "PAAm35mer.pdb"]
PDB_TEMPLATE = ["./pdbtemplates/PAAm25mer.pdb", "./pdbtemplates/PAAm30mer.pdb", "./pdbtemplates/PAAm35mer.pdb", "./pdbtemplates/PAAm40mer.pdb", "./pdbtemplates/PAAm45mer.pdb", "./pdbtemplates/PAAm50mer.pdb"]