"""
Visualisation utilities for the polymer network

show_cylinders() draws:
  - Chain line segments (each chain a random colour)
  - Reactive sites as scatter points (same colour as their chain)
  - BIS crosslink bonds as dashed grey lines between bonded site pairs
  - Chains that participate in cycles drawn with thicker lines
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import random


def random_hex_color():
    return f'#{random.randint(0, 0xFFFFFF):06x}'


def show_cylinders(network, show_bonds=True, highlight_cycles=True):
    """
    Visualise the full polymer network in 3D

    Args:
        network: CylinderNetwork to display.
        show_bonds: If True, draw dashed lines between bonded reactive sites.
        highlight_cycles: If True, draw chains that participate in any cycle
                          with thicker lines
    """
    if not network.cylinders:
        print("No cylinders to show.")
        return

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    # One random colour per chain, consistent across all drawing steps
    colors = [random_hex_color() for _ in network.cylinders]

    # ------------------------------------------------------------------
    # Chain line segments
    # ------------------------------------------------------------------
    segments = np.array([[cyl.start, cyl.end] for cyl in network.cylinders])

    cycle_chain_ids = set()
    if highlight_cycles:
        for cycle in network.find_cycles():
            cycle_chain_ids.update(cycle)

    linewidths = [
        3.0 if i in cycle_chain_ids else 1.5
        for i in range(len(network.cylinders))
    ]

    lc = Line3DCollection(segments, colors=colors, linewidths=linewidths)
    ax.add_collection(lc)

    # ------------------------------------------------------------------
    # Reactive sites — same color as their chain
    # ------------------------------------------------------------------
    for cyl, color in zip(network.cylinders, colors):
        if len(cyl.linking_points) > 0:
            sites = cyl.linking_points
            ax.scatter(
                sites[:, 0], sites[:, 1], sites[:, 2],
                color=color, s=15, zorder=5, alpha=0.75
            )

    # ------------------------------------------------------------------
    # BIS crosslink bonds — dashed grey lines between bonded site pairs
    # ------------------------------------------------------------------
    if show_bonds and network.bonds:
        bond_segs = []
        for (chain_a, site_a, chain_b, site_b) in network.bonds:
            pos_a = network.cylinders[chain_a].linking_points[site_a]
            pos_b = network.cylinders[chain_b].linking_points[site_b]
            bond_segs.append([pos_a, pos_b])

        bond_lc = Line3DCollection(
            bond_segs,
            colors=['#777777'] * len(bond_segs),
            linewidths=1.0,
            linestyles='dashed',
            alpha=0.55
        )
        ax.add_collection(bond_lc)

    # ------------------------------------------------------------------
    # Axis limits — use box dimensions if available, else read network data
    # ------------------------------------------------------------------
    if hasattr(network, 'box_dims') and network.box_dims is not None:
        ax.set_xlim(0, network.box_dims[0])
        ax.set_ylim(0, network.box_dims[1])
        ax.set_zlim(0, network.box_dims[2])
    else:
        all_points = segments.reshape(-1, 3)
        ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
        ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
        ax.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(
        f'Polymer Network — {network.num_chains()} chains, '
        f'{len(network.bonds)} bonds, '
        f'cycle rank {network.cycle_rank()}'
    )

    # Legend entries for the thicker-line convention
    if cycle_chain_ids and highlight_cycles:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='grey', linewidth=1.5, label='Chain'),
            Line2D([0], [0], color='grey', linewidth=3.0, label='Chain in cycle'),
            Line2D([0], [0], color='#777777', linewidth=1.0,
                   linestyle='dashed', label='BIS crosslink'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()