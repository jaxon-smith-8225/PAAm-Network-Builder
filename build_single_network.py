"""
Entry point for polymer network creation

All algorithm logic lives in find_structure.py. This script constructs the
config dictionary from command-line arguments (or config.py defaults) and
hands it off

Usage examples
--------------
# Run with all defaults
python main.py

# Low crosslink density, no percolation requirement
python main.py --crosslink-density 0.01 --no-percolation

# Denser gel, more chains, specific BIS:AAm ratio equivalent
python main.py --crosslink-density 0.05 --volume-fraction 0.08 --chains 30

# Parallel workers explicitly set
python main.py --workers 4
"""

import time
import argparse

from network_builder.find_structure import find_structure
from network_builder.viz import show_cylinders
from network_builder.config import (
    DEFAULT_CHAIN_COUNT,
    BATCH_SIZE,
    MAX_ITERATIONS,
    TARGET_CROSSLINK_DENSITY,
    POLYMER_VOLUME_FRACTION,
    REQUIRE_PERCOLATION,
    TARGET_LOOPS,
    PDB_TEMPLATE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a crosslinked polymer network via pack-and-bond."
    )
    parser.add_argument(
        '--crosslink-density', type=float, default=TARGET_CROSSLINK_DENSITY,
        help=f'Target bonded fraction of reactive sites (default: {TARGET_CROSSLINK_DENSITY})'
    )
    parser.add_argument(
        '--volume-fraction', type=float, default=POLYMER_VOLUME_FRACTION,
        help=f'Polymer volume fraction in box (default: {POLYMER_VOLUME_FRACTION})'
    )
    parser.add_argument(
        '--chains', type=int, default=DEFAULT_CHAIN_COUNT,
        help=f'Target number of chains (default: {DEFAULT_CHAIN_COUNT})'
    )
    parser.add_argument(
        '--target-loops', type=int, default=TARGET_LOOPS,
        help=f'Minimum cycle rank required (default: {TARGET_LOOPS})'
    )
    parser.add_argument(
        '--percolation', action='store_true',
        help='Require the network to percolate before accepting (off by default)'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--pdb', type=str, nargs='+', default=PDB_TEMPLATE,
        help='One or more PDB template files (space-separated)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = {
        'pdb_template':            args.pdb,
        'target_chains':            args.chains,
        'batch_size':               BATCH_SIZE,
        'max_iterations':           MAX_ITERATIONS,
        'target_crosslink_density': args.crosslink_density,
        'polymer_volume_fraction':  args.volume_fraction,
        'require_percolation':      args.percolation,
        'target_loops':             args.target_loops,
    }

    start = time.time()
    network, stats = find_structure(cfg, num_workers=args.workers)
    elapsed = time.time() - start

    if network is None:
        print("\nFailed to generate a network meeting all criteria.")
        print("Try relaxing --crosslink-density, --target-loops, or adding --no-percolation.")
        return

    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"  Chains:            {stats['chains']}")
    print(f"  Bonds:             {stats['bonds']}")
    print(f"  Crosslink density: {stats['crosslink_density'] * 100:.2f}%")
    print(f"  Cycle rank:        {stats['cycle_rank']}")
    print(f"  Percolated:        {stats['percolated']}")
    print(f"  Pack iterations:   {stats['pack_iterations']}")
    print(f"  Bond iterations:   {stats['bond_iterations']}")
    print()
    network.describe_cycles()

    show_cylinders(network)


if __name__ == "__main__":
    main()