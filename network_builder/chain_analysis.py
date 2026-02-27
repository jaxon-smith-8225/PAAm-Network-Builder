"""Functions for analyzing polymer chains and creating cylinder representations"""

import numpy as np
from .models import ChainCylinder
from .config import CHAIN_RADIUS


def scan_chain(universe):
    """
    Scan a polymer chain for reactive sites (amide groups)
    
    Identifies carbon atoms with the pattern: C-O, C-N(H2), where C is also
    bonded to a backbone carbon. This represents the amide functional group
    in polyacrylamide.
    
    Args:
        universe: MDAnalysis Universe containing the polymer chain
        
    Returns:
        List of dictionaries, each containing:
            - central_carbon: The reactive carbon atom (C=O in amide)
            - backbone_carbon: Atom for the backbone carbon
            - oxygen: Atom for the oxygen
            - nitrogen: Atom for the nitrogen  
            - hydrogens: List of hydrogen atoms on nitrogen
    """
    # Pre-select all atoms by element once before loop
    all_carbons = universe.select_atoms('element C')
    all_oxygens = universe.select_atoms('element O')
    all_nitrogens = universe.select_atoms('element N')
    all_hydrogens = universe.select_atoms('element H')
    
    # Create dictionaries for lookup by index
    oxygen_dict = {atom.index: atom for atom in all_oxygens}
    nitrogen_dict = {atom.index: atom for atom in all_nitrogens}
    carbon_dict = {atom.index: atom for atom in all_carbons}
    hydrogen_dict = {atom.index: atom for atom in all_hydrogens}
    
    reactive_groups = []
    
    for c_atom in all_carbons:
        bonded = c_atom.bonded_atoms
        
        # Direct dictionary lookup
        bonded_oxygens = [oxygen_dict[idx] for idx in bonded.indices if idx in oxygen_dict]
        bonded_nitrogens = [nitrogen_dict[idx] for idx in bonded.indices if idx in nitrogen_dict]
        
        # Must have exactly 1 oxygen and 1 nitrogen (amide group)
        if len(bonded_oxygens) != 1 or len(bonded_nitrogens) != 1:
            continue
        
        # Find the backbone carbon
        bonded_carbons = [carbon_dict[idx] for idx in bonded.indices if idx in carbon_dict]
        if len(bonded_carbons) != 1:
            continue
        
        # Check that nitrogen has exactly 2 hydrogens (primary amide)
        n_atom = bonded_nitrogens[0]
        n_bonded = n_atom.bonded_atoms
        n_hydrogens = [hydrogen_dict[idx] for idx in n_bonded.indices if idx in hydrogen_dict]
        
        if len(n_hydrogens) != 2:
            continue
        
        # Found a valid reactive site
        reactive_groups.append({
            'central_carbon': c_atom,
            'backbone_carbon': bonded_carbons[0],
            'oxygen': bonded_oxygens[0],
            'nitrogen': n_atom,
            'hydrogens': list(n_hydrogens)
        })
    
    return reactive_groups


def create_chain_cylinder(chain_atoms, radius=CHAIN_RADIUS):
    """
    Create cylindrical bounding volume for a polymer chain
    
    Args:
        chain_atoms: MDAnalysis AtomGroup representing the chain
        radius: Radius of the cylinder
        
    Returns:
        ChainCylinder object
    """
    # Find terminal carbons by their unique name CTS
    terminal_carbons = chain_atoms.select_atoms('name CTS')
    
    if len(terminal_carbons) == 2:
        start_point = terminal_carbons[0].position.copy()
        end_point = terminal_carbons[1].position.copy()
    elif len(terminal_carbons) < 2:
        # Fallback: use first and last atoms (for BIS or partial structures)
        start_point = chain_atoms[0].position.copy()
        end_point = chain_atoms[-1].position.copy()
    else:
        raise ValueError(
            f"Expected exactly 2 CTS atoms, found {len(terminal_carbons)}. "
            f"Check your PDB template."
        )
    
    # Scan for reactive sites
    # Create a universe from the atoms if needed
    if hasattr(chain_atoms, 'universe'):
        universe = chain_atoms.universe
    else:
        # Assume chain_atoms is already a universe
        universe = chain_atoms
    
    reactive_sites = scan_chain(universe)
    reactive_coords = np.array([
        site['central_carbon'].position 
        for site in reactive_sites
    ])
    
    return ChainCylinder(start_point, end_point, reactive_coords, radius)