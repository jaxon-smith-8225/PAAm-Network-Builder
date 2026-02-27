"""
Pack-and-bond polymer network generation

Iterative pack-and-react approach:

  For each worker:
    1. Compute box dimensions from target chain count and volume fraction
    2. Repeat until stopping criteria are met:
       a. Pack a batch of cylinders randomly into the box (collision checked)
       b. Run a bonding scan across all cylinder pairs
       c. Remove newly added cylinders that formed no bonds (sol fraction)
       d. Check crosslink density, percolation, chain count, and cycle rank
    3. Return the network and stats if criteria are satisfied

Multiple workers run in parallel (ProcessPoolExecutor), each with an
independent random seed. The first worker to succeed wins!
"""

import random
import numpy as np
import MDAnalysis as mda
import warnings
warnings.filterwarnings('ignore')
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from .models import CylinderNetwork
from .chain_analysis import create_chain_cylinder
from .packing import compute_box_dimensions, pack_batch, build_grid
from .bonding import find_candidate_bonds, filter_by_orientation, resolve_conflicts, apply_bonds
from .config import (
    PDB_TEMPLATE,
    DEFAULT_CHAIN_COUNT,
    BATCH_SIZE,
    MAX_ITERATIONS,
    TARGET_CROSSLINK_DENSITY,
    POLYMER_VOLUME_FRACTION,
    REQUIRE_PERCOLATION,
    TARGET_LOOPS,
)


def _bonds_remaining(network, target_density):
    """
    Compute how many more bonds can be formed before hitting the density target

    Each bond commits two sites, so max new bonds = (target_sites - bonded_sites) / 2
    Returns 0 if the target is already met or exceeded.
    """
    total_sites  = sum(len(c.linking_points) for c in network.cylinders)
    target_sites = int(total_sites * target_density)
    current_bonded = len(network.bonded_sites)
    return max(0, (target_sites - current_bonded) // 2)


def build_network_attempt(cfg, seed):
    """
    Run one full pack-and-react attempt with the given config and seed

    Args:
        cfg: Dict with keys:
            pdb_template, target_chains, batch_size, max_iterations,
            target_crosslink_density, polymer_volume_fraction,
            require_percolation, target_loops
        seed: Integer random seed for reproducibility

    Returns:
        (network, stats_dict) if all stopping criteria are met, else None
        stats_dict contains: crosslink_density, cycle_rank, percolated,
        pack_iterations, bond_iterations, chains, bonds
    """
    random.seed(seed)
    np.random.seed(seed)

    template_cylinders = [
        create_chain_cylinder(
            mda.Universe(pdb, context='default', to_guess=['elements', 'bonds']).atoms
        )
        for pdb in cfg['pdb_template']
    ]

    box_dims = compute_box_dimensions(
        template_cylinders,
        cfg['target_chains'],
        cfg['polymer_volume_fraction']
    )

    network = CylinderNetwork(box_dims)

    # Pack chains until the target count is reached.
    pack_iterations = 0
    for iteration in range(cfg['max_iterations']):
        pack_iterations += 1

        if network.num_chains() >= cfg['target_chains']:
            break

        grid = build_grid(network, box_dims)

        remaining = cfg['target_chains'] - network.num_chains()
        batch = min(cfg['batch_size'], remaining)

        new_cyls = pack_batch(network, template_cylinders, batch, box_dims, grid)

        if not new_cyls:
            continue

        start_id = network.num_chains()
        network.add_cylinders(new_cyls)

        max_bonds = max(1, _bonds_remaining(network, cfg['target_crosslink_density'] * 0.5))
        candidates = find_candidate_bonds(network.cylinders, network.bonded_sites)
        candidates = filter_by_orientation(candidates, network.cylinders)
        accepted   = resolve_conflicts(candidates, max_bonds=max_bonds, existing_pairs=network.bonded_chain_pairs)
        apply_bonds(network, accepted)

        network.remove_unbonded_new(start_id)

    # Chain count now fixed. Run bonding-only passes, each forming a small increment of bonds, until all criteria are met.
    bond_iterations = 0
    for iteration in range(cfg['max_iterations']):
        bond_iterations += 1

        density_ok     = cfg['target_crosslink_density']+0.0015 >= network.crosslink_density >= cfg['target_crosslink_density']-0.0015
        percolation_ok = (not cfg['require_percolation']) or network.is_percolated()
        loops_ok       = network.cycle_rank() >= cfg['target_loops']

        if density_ok and percolation_ok and loops_ok:
            stats = {
                'crosslink_density': network.crosslink_density,
                'cycle_rank':        network.cycle_rank(),
                'percolated':        network.is_percolated(),
                'pack_iterations':   pack_iterations,
                'bond_iterations':   bond_iterations,
                'chains':            network.num_chains(),
                'bonds':             len(network.bonds),
            }
            return network, stats

        max_bonds = min(cfg['batch_size'], _bonds_remaining(network, cfg['target_crosslink_density']))
        if max_bonds == 0:
            # Density target reached but other criteria not yet met
            break

        candidates = find_candidate_bonds(network.cylinders, network.bonded_sites)
        candidates = filter_by_orientation(candidates, network.cylinders)
        accepted   = resolve_conflicts(candidates, max_bonds=max_bonds)

        if not accepted:
            break  # Network is saturated

        apply_bonds(network, accepted)

    return None


def find_structure(cfg=None, num_workers=None):
    """
    Run parallel pack-and-bond attempts until one meets all criteria

    Workers run independently with different random seeds. The first
    successful result is returned and all other futures are cancelled

    Args:
        cfg: Configuration dict. If None, defaults from config.py are used
             Keys: pdb_template, target_chains, batch_size, max_iterations,
             target_crosslink_density, polymer_volume_fraction,
             require_percolation, target_loops.
        num_workers: Number of parallel workers (default: CPU count)

    Returns:
        (network, stats_dict) for the first successful structure found,
        or (None, None) if all workers exhaust their iterations
    """
    if cfg is None:
        cfg = {
            'pdb_templates':            [PDB_TEMPLATE],
            'target_chains':            DEFAULT_CHAIN_COUNT,
            'batch_size':               BATCH_SIZE,
            'max_iterations':           MAX_ITERATIONS,
            'target_crosslink_density': TARGET_CROSSLINK_DENSITY,
            'polymer_volume_fraction':  POLYMER_VOLUME_FRACTION,
            'require_percolation':      REQUIRE_PERCOLATION,
            'target_loops':             TARGET_LOOPS,
        }

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    rng = random.Random()

    def next_seed():
        return rng.randint(0, 2 ** 32 - 1)

    print(f"Starting search with {num_workers} workers")
    print(
        f"Targets: {cfg['target_chains']} chains | "
        f"{cfg['target_crosslink_density'] * 100:.1f}% crosslink density | "
        f"cycle rank >= {cfg['target_loops']} | "
        f"{'percolation required' if cfg['require_percolation'] else 'no percolation requirement'}"
    )
    print(f"Box volume fraction: {cfg['polymer_volume_fraction'] * 100:.1f}%")

    attempt = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(build_network_attempt, cfg, next_seed()): None
            for _ in range(num_workers)
        }

        while futures:
            for future in as_completed(futures):
                futures.pop(future)
                attempt += 1
                result = future.result()

                if result is not None:
                    for f in futures:
                        f.cancel()
                    network, stats = result
                    print(f"\nSuccess on attempt {attempt}")
                    return network, stats

                print(f"Attempt {attempt} did not converge, retrying...")
                futures[executor.submit(build_network_attempt, cfg, next_seed())] = None
                break  # back to as_completed

    print("Warning: no attempt produced a valid network.")
    return None, None