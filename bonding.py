"""
Bonding scan utilities for the pack-and-bond algorithm.

All functions here are stateless: they take the current cylinder list and
bonded-site set as arguments and return candidate or accepted bond lists.
This makes them easy to unit-test independently of the network object

Bond format throughout this module:
    (chain_id_a, site_idx_a, chain_id_b, site_idx_b)

Candidate format adds a distance field:
    (chain_id_a, site_idx_a, chain_id_b, site_idx_b, distance)
"""

import numpy as np
from scipy.spatial import cKDTree
from config import BOND_SEARCH_RADIUS, MIN_RADIAL_DOT_PRODUCT


def find_candidate_bonds(cylinders, bonded_sites):
    """
    Find all free reactive site pairs within BOND_SEARCH_RADIUS using a KD-tree

    All free reactive site coordinates across all cylinders are stacked into a
    single array and passed to cKDTree.query_pairs(), which returns all pairs
    within the search radius in O(n log n). Each result is tagged with the
    originating chain ID and site index so the caller can look up geometry

    Sites already in bonded_sites are excluded before the tree is built, so
    the tree only contains sites that are still available.

    Args:
        cylinders: List of ChainCylinder objects.
        bonded_sites: Set of (chain_id, site_idx) pairs already committed.

    Returns:
        List of (chain_id_a, site_a, chain_id_b, site_b, distance) tuples.
        Only inter-chain pairs are returned (sites on the same chain are
        excluded because a chain cannot crosslink to itself via BIS).
    """
    if not cylinders:
        return []

    # Build flat arrays of all free site coordinates and their chain/site IDs
    all_coords = []
    all_ids    = []   # (chain_id, site_idx) for each row in all_coords

    for chain_id, cyl in enumerate(cylinders):
        for site_idx, coord in enumerate(cyl.linking_points):
            if (chain_id, site_idx) not in bonded_sites:
                all_coords.append(coord)
                all_ids.append((chain_id, site_idx))

    if len(all_coords) < 2:
        return []

    coords_arr = np.array(all_coords)   # shape (N, 3)
    tree = cKDTree(coords_arr)

    # query_pairs returns all pairs (i, j) with i < j and dist <= radius
    pairs = tree.query_pairs(BOND_SEARCH_RADIUS, output_type='ndarray')

    if len(pairs) == 0:
        return []

    candidates = []
    for (i, j) in pairs:
        chain_a, site_a = all_ids[i]
        chain_b, site_b = all_ids[j]

        # Exclude same-chain pairs --> a chain cannot bond to itself
        if chain_a == chain_b:
            continue

        dist = float(np.linalg.norm(coords_arr[i] - coords_arr[j]))
        candidates.append((chain_a, site_a, chain_b, site_b, dist))

    return candidates


def filter_by_orientation(candidates, cylinders):
    """
    Filter bond candidates by radial orientation compatibility

    A bond is accepted when the outward radial vectors from each chain's axis
    at the two reactive site positions have a dot product <= MIN_RADIAL_DOT_PRODUCT
    A value of 0.5 means sites within roughly 120 degrees of facing each other
    pass the filter Sites that lie exactly on their chain's axis have no meaningful 
    radial direction and are accepted unconditionally

    Args:
        candidates: List of (chain_id_a, site_a, chain_id_b, site_b, dist)
        cylinders: List of ChainCylinder objects

    Returns:
        Filtered subset of candidates
    """
    filtered = []

    for (chain_a, site_a, chain_b, site_b, dist) in candidates:
        cyl_a = cylinders[chain_a]
        cyl_b = cylinders[chain_b]

        # Radial vector: from axis projection point to reactive site
        t_a = cyl_a.get_parameter_t(site_a)
        axis_pt_a = cyl_a.start + t_a * cyl_a.axis
        radial_a = cyl_a.linking_points[site_a] - axis_pt_a

        t_b = cyl_b.get_parameter_t(site_b)
        axis_pt_b = cyl_b.start + t_b * cyl_b.axis
        radial_b = cyl_b.linking_points[site_b] - axis_pt_b

        norm_a = np.linalg.norm(radial_a)
        norm_b = np.linalg.norm(radial_b)

        # Site on axis -> no meaningful orientation, accept unconditionally
        if norm_a < 1e-6 or norm_b < 1e-6:
            filtered.append((chain_a, site_a, chain_b, site_b, dist))
            continue

        dot = np.dot(radial_a / norm_a, radial_b / norm_b)
        if dot <= MIN_RADIAL_DOT_PRODUCT:
            filtered.append((chain_a, site_a, chain_b, site_b, dist))

    return filtered


def resolve_conflicts(candidates, max_bonds=None, existing_pairs=None):
    """
    Greedily assign bonds so each reactive site participates in at most one

    Sorts candidates by distance (shortest first, closest geometry preferred),
    then assigns bonds while tracking which sites are already claimed

    Args:
        candidates: List of (chain_id_a, site_a, chain_id_b, site_b, dist)
        max_bonds: If set, stop accepting bonds after this many are assigned
                   Used to hit a crosslink density target without over-bonding

    Returns:
        List of accepted (chain_id_a, site_a, chain_id_b, site_b) tuples
        with no site appearing more than once
    """
    sorted_candidates = sorted(candidates, key=lambda x: x[4])

    claimed      = set()
    bonded_pairs = set(existing_pairs or [])
    accepted     = []

    for (chain_a, site_a, chain_b, site_b, _) in sorted_candidates:
        if max_bonds is not None and len(accepted) >= max_bonds:
            break
        if (chain_a, site_a) in claimed or (chain_b, site_b) in claimed:
            continue

        pair = frozenset((chain_a, chain_b))
        if pair in bonded_pairs:
            continue

        accepted.append((chain_a, site_a, chain_b, site_b))
        claimed.add((chain_a, site_a))
        claimed.add((chain_b, site_b))
        bonded_pairs.add(pair)

    return accepted


def apply_bonds(network, accepted_bonds):
    """
    Record accepted bonds in the network's bond list and bonded_sites set

    Args:
        network: CylinderNetwork to update in place
        accepted_bonds: List of (chain_id_a, site_a, chain_id_b, site_b)
    """
    for (chain_a, site_a, chain_b, site_b) in accepted_bonds:
        network.bonds.append((chain_a, site_a, chain_b, site_b))
        network.bonded_sites.add((chain_a, site_a))
        network.bonded_sites.add((chain_b, site_b))