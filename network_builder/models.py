"""Data models for polymer chain cylinders and network structure"""

import numpy as np
import networkx as nx
from .geometry_utils import (
    rotate_points,
    angle_between_vectors,
    cylinders_collide
)
from .config import CHAIN_RADIUS


class ChainCylinder:
    """
    Represents a cylindrical bounding volume for a polymer chain

    The cylinder is defined by start and end points (terminal atoms) and
    contains reactive site coordinates along the chain
    """

    def __init__(self, start_point, end_point, reactive_coords, radius=CHAIN_RADIUS):
        self.points = np.vstack((start_point, end_point, reactive_coords))
        self.radius = radius
        self.start = self.points[0]
        self.end = self.points[1]
        self.linking_points = self.points[2:]
        self.all_transformations = []
        self._cached_axis = None
        self._cached_length = None

    def _refresh_points(self):
        self.start = self.points[0]
        self.end = self.points[1]
        self.linking_points = self.points[2:]
        self._cached_axis = None
        self._cached_length = None

    @property
    def axis(self):
        if self._cached_axis is None:
            self._cached_axis = self.end - self.start
        return self._cached_axis

    @property
    def length(self):
        if self._cached_length is None:
            self._cached_length = np.linalg.norm(self.axis)
        return self._cached_length

    @property
    def relative_linking_points(self):
        return self.linking_points - self.start

    def copy(self):
        new_cyl = ChainCylinder.__new__(ChainCylinder)
        new_cyl.points = self.points.copy()
        new_cyl.radius = self.radius
        new_cyl.start = new_cyl.points[0]
        new_cyl.end = new_cyl.points[1]
        new_cyl.linking_points = new_cyl.points[2:]
        new_cyl.all_transformations = []
        new_cyl._cached_axis = None
        new_cyl._cached_length = None
        return new_cyl

    def get_parameter_t(self, site_idx):
        if self.length < 1e-10:
            return 0.0
        relative_point = self.relative_linking_points[site_idx]
        projection = np.dot(relative_point, self.axis) / self.length
        return projection / self.length

    def align(self, target_vec):
        rotation_axis = np.cross(self.axis, target_vec)
        rotation_angle = angle_between_vectors(self.axis, target_vec)
        self.points = rotate_points(self.points, rotation_axis, rotation_angle, self.start)
        self._refresh_points()
        self.all_transformations.append(('align', target_vec))

    def translate(self, displacement_vector):
        self.points += displacement_vector
        self._refresh_points()
        self.all_transformations.append(('translate', displacement_vector))

    def twist(self, site_idx, angle):
        t = self.get_parameter_t(site_idx)
        rotation_center = self.start + t * self.axis
        self.points = rotate_points(
            self.points,
            rotation_axis=self.axis,
            angle_degrees=angle,
            center=rotation_center
        )
        self._refresh_points()
        self.all_transformations.append(('twist', site_idx, angle))

    def calculate_alignment_angle(self, site_idx, target_position):
        t = self.get_parameter_t(site_idx)
        axis_point = self.start + t * self.axis
        current_radial = self.linking_points[site_idx] - axis_point
        desired_direction = target_position - axis_point
        axis_normalized = self.axis / self.length
        current_proj = current_radial - np.dot(current_radial, axis_normalized) * axis_normalized
        desired_proj = desired_direction - np.dot(desired_direction, axis_normalized) * axis_normalized
        if np.linalg.norm(current_proj) < 1e-6 or np.linalg.norm(desired_proj) < 1e-6:
            return 0.0
        angle = angle_between_vectors(current_proj, desired_proj)
        cross = np.cross(current_proj, desired_proj)
        if np.dot(cross, axis_normalized) < 0:
            angle = -angle
        return angle


class CylinderNetwork:
    """
    Manages a crosslinked polymer network built by the pack-and-bond algorithm

    Connectivity is stored as a flat bond list rather than a per-chain adjacency
    list, which makes graph construction, cycle analysis, and ID remapping after
    cylinder removal all straightforward

    Attributes
    ----------
    box_dims : np.ndarray shape (3,)
        Simulation box [Lx, Ly, Lz] in Angstroms
    cylinders : list[ChainCylinder]
        All chains currently in the network. A chain's ID is its index here
    bonds : list[tuple]
        Each entry is (chain_id_a, site_idx_a, chain_id_b, site_idx_b)
    bonded_sites : set[tuple]
        (chain_id, site_idx) pairs that are already committed to a bond
        Used during the bonding scan to skip already-bonded sites
    """

    def __init__(self, box_dims):
        """
        Initialise an empty network within a box

        Args:
            box_dims: Array-like [Lx, Ly, Lz] in Angstroms
        """
        self.box_dims = np.array(box_dims, dtype=float)
        self.cylinders = []
        self.bonds = []            # (chain_a, site_a, chain_b, site_b)
        self.bonded_sites = set()  # (chain_id, site_idx)

    # Basic accessors

    def num_chains(self):
        """Total number of chains currently in the network"""
        return len(self.cylinders)

    # Cylinder management

    def add_cylinders(self, cylinders):
        """
        Bulk-add a list of ChainCylinder objects without bonding them.

        Cylinders are appended in order; each one's chain ID is its resulting
        index in self.cylinders.

        Args:
            cylinders: List of ChainCylinder objects.
        """
        self.cylinders.extend(cylinders)

    def remove_unbonded_new(self, start_id):
        """
        Remove cylinders added since start_id that formed no bonds this round.

        After each bonding scan, newly placed cylinders that did not bond to
        anything represent the sol fraction and are discarded. All cylinder IDs
        and bond indices are remapped so the network stays self-consistent.

        Args:
            start_id: Index of the first cylinder added in the current batch.

        Returns:
            int: Number of cylinders removed.
        """
        bonded_chain_ids = set()
        for (a, _, b, _) in self.bonds:
            bonded_chain_ids.add(a)
            bonded_chain_ids.add(b)

        to_remove = {
            i for i in range(start_id, len(self.cylinders))
            if i not in bonded_chain_ids
        }

        if not to_remove:
            return 0

        # Build old -> new ID mapping skipping removed cylinders
        old_to_new = {}
        new_cylinders = []
        new_id = 0
        for old_id, cyl in enumerate(self.cylinders):
            if old_id not in to_remove:
                old_to_new[old_id] = new_id
                new_cylinders.append(cyl)
                new_id += 1

        # Remap all references to cylinder IDs
        self.bonds = [
            (old_to_new[a], sa, old_to_new[b], sb)
            for (a, sa, b, sb) in self.bonds
        ]
        self.bonded_sites = {
            (old_to_new[cid], sid)
            for (cid, sid) in self.bonded_sites
        }
        self.cylinders = new_cylinders

        return len(to_remove)

    def check_collision(self, new_cylinder, skip_cylinders=None):
        """
        Check if new_cylinder collides with any existing cylinder

        Args:
            new_cylinder: ChainCylinder to test
            skip_cylinders: Optional list of cylinders to skip

        Returns:
            True if a collision is detected
        """
        skip = {id(c) for c in (skip_cylinders or [])}
        for cyl in self.cylinders:
            if id(cyl) in skip:
                continue
            if cylinders_collide(new_cylinder, cyl):
                return True
        return False

    # Network properties

    @property
    def crosslink_density(self):
        """Fraction of reactive sites that are bonded (0.0 – 1.0)."""
        total = sum(len(cyl.linking_points) for cyl in self.cylinders)
        if total == 0:
            return 0.0
        return len(self.bonded_sites) / total
    
    @property
    def bonded_chain_pairs(self):
        """Set of frozensets of chain ID pairs that already share a crosslink."""
        return {frozenset((a, b)) for (a, _, b, _) in self.bonds}

    def is_percolated(self):
        """
        Check whether the network has formed a spanning cluster (gel point).

        Either of two criteria is sufficient:
          1. The largest connected component contains >= 60% of all chains.
          2. The centroids of chains in the largest component span >= 60% of
             the simulation box in at least one dimension.

        Returns:
            bool
        """
        if not self.bonds or not self.cylinders:
            return False

        G = self.build_graph()
        components = list(nx.connected_components(G))
        largest = max(components, key=len)

        # Criterion 1: majority of chains are connected
        if len(largest) >= 0.6 * len(self.cylinders):
            return True

        # Criterion 2: spatial extent across the box
        if len(largest) >= 2:
            centroids = np.array([
                (self.cylinders[i].start + self.cylinders[i].end) / 2
                for i in largest
            ])
            span = centroids.max(axis=0) - centroids.min(axis=0)
            if np.any(span >= self.box_dims * 0.6):
                return True

        return False

    def cycle_rank(self):
        """
        Number of independent cycles (circuit rank / first Betti number)

        This is the physically meaningful loop count — the number of BIS
        crosslinks beyond those needed to span the network as a tree
        Matches what the old construction counter was counting

        Formula:  edges - nodes + connected_components
        """
        G = self.build_graph()
        n = G.number_of_nodes()
        e = G.number_of_edges()
        c = nx.number_connected_components(G)
        return max(0, e - n + c)

    # Graph and cycle analysis

    def build_graph(self):
        """
        Build an undirected NetworkX graph from the bond list

        Nodes are chain IDs. Edges are BIS crosslinks. Multiple bonds between
        the same chain pair collapse to a single edge (nx.Graph deduplicates),
        so simple_cycles does not over-count

        Returns:
            nx.Graph
        """
        G = nx.Graph()
        G.add_nodes_from(range(len(self.cylinders)))
        for (a, _, b, _) in self.bonds:
            G.add_edge(a, b)
        return G

    def find_cycles(self):
        """
        Find all simple cycles in the network topology

        Returns:
            List of cycles (each a list of chain IDs), sorted shortest first
        """
        G = self.build_graph()
        raw = list(nx.simple_cycles(G))
        cycles = [c for c in raw if len(c) > 2]
        return sorted(cycles, key=len)

    def describe_cycles(self):
        """Print a human-readable summary of cycles and cycle rank"""
        cycles = self.find_cycles()
        rank = self.cycle_rank()
        if not cycles:
            print("No cycles found in network.")
            return
        print(f"Cycle rank (independent loops): {rank}")
        print(f"Simple cycles (all closed paths): {len(cycles)}")
        for i, cycle in enumerate(cycles):
            closed = cycle + [cycle[0]]
            path_str = " -> ".join(str(c) for c in closed)
            print(f"  Cycle {i+1} (length {len(cycle)}): {path_str}")

    def __repr__(self):
        return (
            f"CylinderNetwork("
            f"chains={self.num_chains()}, "
            f"bonds={len(self.bonds)}, "
            f"crosslink_density={self.crosslink_density:.3f}, "
            f"cycle_rank={self.cycle_rank()})"
        )