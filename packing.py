"""
Cylinder packing utilities for the pack-and-bond algorithm

Responsible for placing ChainCylinder objects randomly into the simulation
box with collision checking

Orientation sampling uses scipy Rotation.random() for a proper uniform
distribution on SO(3), avoiding the bias of the old align() + random
vector approach
"""

import math
from collections import defaultdict
import random

import numpy as np
from scipy.spatial.transform import Rotation

from geometry_utils import cylinders_collide, segment_segment_distance
from config import CHAIN_RADIUS


# Spatial grid

class SpatialGrid:
    """
    Uniform grid accelerator for cylinder collision detection

    Cylinders are registered in every cell their axis segment passes through
    (via 3-D DDA traversal). Collision queries retrieve only the cylinders
    in cells neighbouring the query cylinder's axis, then call the exact
    cylinders_collide check on that small candidate set
    """

    def __init__(self, box_dims, cell_size=None):
        """
        Args:
            box_dims: np.ndarray([Lx, Ly, Lz])
            cell_size: Grid cell side length. Defaults to 2 * CHAIN_RADIUS,
                       which guarantees that any two colliding cylinders share
                       at least one cell
        """
        self.cell_size = cell_size or (2.0 * CHAIN_RADIUS)
        self.box_dims  = np.array(box_dims)
        self._grid     = defaultdict(list)   # cell tuple -> [ChainCylinder]

    def _cells_for_segment(self, start, end):
        """
        Return the set of grid cells that the segment start->end passes through

        Sampling approach: step along the segment in increments
        of half the cell size, recording each occupied cell. This is
        conservative (may include a few extra cells at oblique angles) but
        cheap and correct
        """
        length = np.linalg.norm(end - start)
        if length < 1e-10:
            return {self._cell(start)}

        n_steps = max(2, int(length / (self.cell_size * 0.5)) + 1)
        cells = set()
        for k in range(n_steps + 1):
            t = k / n_steps
            point = start + t * (end - start)
            cells.add(self._cell(point))
        return cells

    def _cell(self, point):
        """Convert a 3D point to integer grid cell indices"""
        return tuple((point / self.cell_size).astype(int))

    def _neighbour_cells(self, cells):
        """
        Return all cells within 1 step (26-neighbourhood) of any given cell

        This ensures that two cylinders whose axes come within CHAIN_RADIUS
        of each other will always share at least one neighbouring cell
        """
        neighbours = set()
        for (cx, cy, cz) in cells:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbours.add((cx + dx, cy + dy, cz + dz))
        return neighbours

    def add(self, cylinder):
        """Register a cylinder in the grid"""
        for cell in self._cells_for_segment(cylinder.start, cylinder.end):
            self._grid[cell].append(cylinder)

    def candidates(self, cylinder):
        """Return the set of cylinders that could collide with the given one"""
        cells = self._cells_for_segment(cylinder.start, cylinder.end)
        nearby = self._neighbour_cells(cells)
        seen = set()
        result = []
        for cell in nearby:
            for cyl in self._grid.get(cell, []):
                if id(cyl) not in seen:
                    seen.add(id(cyl))
                    result.append(cyl)
        return result

    def collides(self, cylinder):
        """
        Return True if cylinder collides with any registered cylinder

        Only cylinders in neighbouring grid cells are tested exactly
        """
        for other in self.candidates(cylinder):
            if cylinders_collide(cylinder, other):
                return True
        return False


# Public API

def compute_box_dimensions(template_cylinders, n_chains, phi):
    """
    Compute cubic simulation box dimensions from chain count and volume fraction

    Box volume is set so that n_chains chains fill fraction phi of the volume

    Args:
        template_cylinder: ChainCylinder whose length is used for volume estimate
        n_chains: Target number of chains
        phi: Polymer volume fraction (e.g. 0.05 for 5%)

    Returns:
        np.ndarray([Lx, Ly, Lz]) in Angstroms
    """
    avg_volume = np.mean([
        math.pi * CHAIN_RADIUS ** 2 * t.length
        for t in template_cylinders
    ])
    box_volume = (n_chains * avg_volume) / phi
    side = box_volume ** (1.0 / 3.0)
    return np.array([side, side, side])


def random_orientation(template_cylinder):
    """
    Return a copy of template_cylinder with a uniformly random orientation

    Samples a rotation uniformly from SO(3) via scipy Rotation.random(),
    then aligns the cylinder axis to the rotated z-axis.

    Args:
        template_cylinder: ChainCylinder to copy and reorient

    Returns:
        New ChainCylinder with random orientation, start at origin
    """
    cyl = template_cylinder.copy()
    target_direction = Rotation.random().apply(np.array([0.0, 0.0, 1.0]))
    cyl.align(target_direction)
    return cyl


def pack_batch(network, template_cylinders, batch_size, box_dims, grid):
    """
    Place up to batch_size cylinders randomly in the box using grid acceleration

    Each attempt:
      1. Copies and randomly reorients the template cylinder
      2. Places its centroid at a uniformly random position in the box
      3. Checks collision via the spatial grid (O(1) average cost)
      4. On success, registers the cylinder in the grid and the batch list

    Cylinders may extend outside the box — no clipping is applied

    Args:
        network: CylinderNetwork (used only for grid initialisation bookkeeping;
                 existing cylinders must already be in grid before calling)
        template_cylinder: ChainCylinder shape template
        batch_size: Maximum number of cylinders to place
        box_dims: np.ndarray([Lx, Ly, Lz])
        grid: SpatialGrid already populated with network.cylinders

    Returns:
        List of successfully placed ChainCylinder objects (length <= batch_size)
    """
    placed = []
    max_attempts = batch_size * 30
    attempts = 0

    while len(placed) < batch_size and attempts < max_attempts:
        attempts += 1

        template = random.choice(template_cylinders)   # <-- random size pick
        cyl = random_orientation(template)

        centroid = (cyl.start + cyl.end) / 2.0
        random_pos = np.random.uniform(np.zeros(3), box_dims)
        cyl.translate(random_pos - centroid)

        if not grid.collides(cyl):
            grid.add(cyl)
            placed.append(cyl)

    return placed


def build_grid(network, box_dims):
    """
    Build a SpatialGrid pre-populated with all cylinders in the network

    Call this once per iteration before calling pack_batch, then pass the
    same grid to pack_batch so newly placed cylinders are registered
    incrementally rather than rebuilding the whole grid each time

    Args:
        network: CylinderNetwork whose cylinders are registered
        box_dims: np.ndarray([Lx, Ly, Lz])

    Returns:
        Populated SpatialGrid
    """
    grid = SpatialGrid(box_dims)
    for cyl in network.cylinders:
        grid.add(cyl)
    return grid