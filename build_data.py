# build_data.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
import os
import numpy as np
import torch
from torch_geometric.data import Data
import itertools
import logging
import random
from typing import List, Tuple, Dict, Any, Optional
from logger import get_logger
from bigsmiles_utils import roundtrip_or_fallback
from config import Configuration   

# Initialize logger
logger = get_logger(__name__)

# Utility Functions
def return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i):
    fg_without_c_i_wash = []
    for fg_with_c in fg_with_c_i:
        for fg_without_c in fg_without_c_i:
            if set(fg_without_c).issubset(set(fg_with_c)):
                fg_without_c_i_wash.append(list(fg_without_c))
    return fg_without_c_i_wash

def return_fg_hit_atom(smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list):
    mol = Chem.MolFromSmiles(smiles)
    hit_at = []
    hit_fg_name = []
    all_hit_fg_at = []
    for i in range(len(fg_with_ca_list)):
        fg_with_c_i = mol.GetSubstructMatches(fg_with_ca_list[i])
        fg_without_c_i = mol.GetSubstructMatches(fg_without_ca_list[i])
        fg_without_c_i_wash = return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i)
        if len(fg_without_c_i_wash) > 0:
            hit_at.append(fg_without_c_i_wash)
            hit_fg_name.append(fg_name_list[i])
            all_hit_fg_at += fg_without_c_i_wash

    sorted_all_hit_fg_at = sorted(all_hit_fg_at, key=lambda fg: len(fg), reverse=True)

    remain_fg_list = []
    for fg in sorted_all_hit_fg_at:
        if fg not in remain_fg_list:
            if len(remain_fg_list) == 0:
                remain_fg_list.append(fg)
            else:
                i = 0
                for remain_fg in remain_fg_list:
                    if set(fg).issubset(set(remain_fg)):
                        break
                    else:
                        i += 1
                if i == len(remain_fg_list):
                    remain_fg_list.append(fg)

    hit_at_wash = []
    hit_fg_name_wash = []
    for j in range(len(hit_at)):
        hit_at_wash_j = []
        for fg in hit_at[j]:
            if fg in remain_fg_list:
                hit_at_wash_j.append(fg)
        if len(hit_at_wash_j) > 0:
            hit_at_wash.append(hit_at_wash_j)
            hit_fg_name_wash.append(hit_fg_name[j])
    return hit_at_wash, hit_fg_name_wash

def getAllBricsBondSubset(BricsBond, max_subsets=10000):
    """
    Generate all possible combinations of BRICS bonds up to a maximum limit.
    Args:
        BricsBond: List of BRICS bonds
        max_subsets: Maximum number of combinations to generate
    Returns:
        List of bond subset combinations
    """
    all_brics_bond_subset = []
    N = len(BricsBond)
    
    # Generate all possible combinations using binary counting
    for i in range(2 ** N):
        brics_bond_subset = []
        for j in range(N):
            if (i >> j) % 2:
                brics_bond_subset.append(BricsBond[j])
        if len(brics_bond_subset) > 0:
            all_brics_bond_subset.append(brics_bond_subset)
        if len(all_brics_bond_subset) > max_subsets:
            logger.warning(f"Reached maximum subset limit of {max_subsets}")
            break
    return all_brics_bond_subset



def get_fragment_atoms(mol, start_atom, exclude=None):
    """Get connected atoms in a fragment."""
    if exclude is None:
        exclude = []
        
    visited = set([start_atom])
    to_visit = set([start_atom])
    exclude = set(exclude)
    
    while to_visit:
        atom_idx = to_visit.pop()
        atom = mol.GetAtomWithIdx(atom_idx)
        
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if (neighbor_idx not in visited and 
                neighbor_idx not in exclude):
                visited.add(neighbor_idx)
                to_visit.add(neighbor_idx)
    
    return list(visited)

def return_brics_leaf_structure(smiles):
    """
    BRICS decomposition focusing on medicinally relevant synthetic breaks.
    Prioritizes breaks at common medicinal chemistry connection points
    like amides, amines, esters, aromatic linkages, and ethers.

    Returns a dictionary with:
    - 'substructure': A dict of fragments indexed by integer keys.
    - 'reaction_centers': A dict containing the atoms where the chosen break occurs.
    - 'brics_bond_types': A list of the chosen break type(s).
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Priority BRICS rules for medicinal chemistry
    priority_breaks = {
        '7': 'amide',      
        '5': 'amine',      
        '6': 'ester',      
        '4': 'aromatic',   
        '3': 'ether'
    }

    # Find all BRICS bonds
    brics_bonds = list(BRICS.FindBRICSBonds(m))
    
    all_brics_substructure_subset = dict()

    if brics_bonds:
        # Evaluate and sort BRICS bonds by defined priority
        prioritized_bonds = []
        for (atom1, atom2), (break_type1, break_type2) in brics_bonds:
            # Determine priority
            p1 = int(break_type1) if break_type1 in priority_breaks else 99
            p2 = int(break_type2) if break_type2 in priority_breaks else 99
            priority = min(p1, p2)

            # Only consider if at least one of the break_types is in priority_breaks
            if priority < 99:
                prioritized_bonds.append((priority, (atom1, atom2), (break_type1, break_type2)))

        # Sort by priority, lowest is highest priority
        prioritized_bonds.sort()

        if prioritized_bonds:
            # Take the top-priority bond
            _, (atom1, atom2), (break_type1, break_type2) = prioritized_bonds[0]
            chosen_break_type = break_type1 if break_type1 in priority_breaks else break_type2

            # Extract fragments after this priority break
            fragment1 = get_fragment_atoms(m, atom1, exclude=[atom2])
            fragment2 = get_fragment_atoms(m, atom2, exclude=[atom1])
            
            substrate_idx = {
                0: fragment1,
                1: fragment2
            }

            # Reaction centers: the atoms at the break
            reaction_centers = {0: [atom1, atom2]}

            # Store chosen BRICS bond type(s)
            # Using a list here, but you could store more detail if needed
            brics_bond_types = [chosen_break_type]

            all_brics_substructure_subset['substructure'] = substrate_idx
            all_brics_substructure_subset['reaction_centers'] = reaction_centers
            all_brics_substructure_subset['brics_bond_types'] = brics_bond_types
        else:
            # No priority breaks found; fallback to whole molecule
            substrate_idx = {0: [x for x in range(m.GetNumAtoms())]}
            all_brics_substructure_subset['substructure'] = substrate_idx
            all_brics_substructure_subset['reaction_centers'] = {}
            all_brics_substructure_subset['brics_bond_types'] = []
    else:
        # No BRICS bonds at all; just return the whole molecule as a single substructure
        substrate_idx = {0: [x for x in range(m.GetNumAtoms())]}
        all_brics_substructure_subset['substructure'] = substrate_idx
        all_brics_substructure_subset['reaction_centers'] = {}
        all_brics_substructure_subset['brics_bond_types'] = []

    return all_brics_substructure_subset


def return_murcko_leaf_structure(smiles):
    m = Chem.MolFromSmiles(smiles)
    res = list(BRICS.FindBRICSBonds(m))

    all_murcko_bond = [set(res[i][0]) for i in range(len(res))]
    all_murcko_substructure_subset = dict()
    all_murcko_atom = []
    for murcko_bond in all_murcko_bond:
        all_murcko_atom = list(set(all_murcko_atom + list(murcko_bond)))

    if len(all_murcko_atom) > 0:
        all_break_atom = dict()
        for murcko_atom in all_murcko_atom:
            murcko_break_atom = []
            for murcko_bond in all_murcko_bond:
                if murcko_atom in murcko_bond:
                    murcko_break_atom += list(set(murcko_bond))
            murcko_break_atom = [x for x in murcko_break_atom if x != murcko_atom]
            all_break_atom[murcko_atom] = murcko_break_atom

        substrate_idx = dict()
        used_atom = []
        for initial_atom_idx, break_atoms_idx in all_break_atom.items():
            if initial_atom_idx not in used_atom:
                neighbor_idx = [initial_atom_idx]
                substrate_idx_i = neighbor_idx
                begin_atom_idx_list = [initial_atom_idx]
                while len(neighbor_idx) != 0:
                    for idx in begin_atom_idx_list:
                        initial_atom = m.GetAtomWithIdx(idx)
                        neighbor_idx = neighbor_idx + [neighbor_atom.GetIdx() for neighbor_atom in
                                                       initial_atom.GetNeighbors()]
                        exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i
                        if idx in all_break_atom.keys():
                            exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i + all_break_atom[idx]
                        neighbor_idx = [x for x in neighbor_idx if x not in exlude_idx]
                        substrate_idx_i += neighbor_idx
                        begin_atom_idx_list += neighbor_idx
                    begin_atom_idx_list = [x for x in begin_atom_idx_list if x not in substrate_idx_i]
                substrate_idx[initial_atom_idx] = substrate_idx_i
                used_atom += substrate_idx_i
            else:
                pass
    else:
        substrate_idx = dict()
        substrate_idx[0] = [x for x in range(m.GetNumAtoms())]
    all_murcko_substructure_subset['substructure'] = substrate_idx
    all_murcko_substructure_subset['substructure_bond'] = all_murcko_bond
    return all_murcko_substructure_subset


def atom_features(atom, use_chirality=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As',
            'Se', 'Br', 'Te', 'I', 'At', '*', 'other'
        ]) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def etype_features(bond, use_chirality=True):
    features = []
    
    # Bond type
    bond_type = bond.GetBondType()
    features.extend([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC
    ])
    
    # Other features
    features.append(bond.GetIsConjugated())
    features.append(bond.IsInRing())
    
    if use_chirality:
        stereo = str(bond.GetStereo())
        features.extend([
            stereo == "STEREONONE",
            stereo == "STEREOANY",
            stereo == "STEREOZ",
            stereo == "STEREOE"
        ])
    
    return features  # This will be a list of boolean values



def construct_mol_graph_from_smiles(smiles, smask):
    """Construct molecular graph from SMILES for 3-edge RGCN (v2: -1 aromatic filter).

    Principle: primary graphs include aromaticity, but aromatic bonds are NOT
    used as relation types. We assign edge_type = -1 for aromatic bonds and
    filter them out inside the model during message passing. Non-aromatic bonds
    map to SINGLE/DOUBLE/TRIPLE (0/1/2). Aromaticity remains in node features.
    """
    try:
        # Allow wildcard atoms (*) by disabling strict sanitization
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Attempt sanitization while gracefully handling errors due to wildcards
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                             catchErrors=True)

        # Use a single aromatic-aware molecule for both features and edges
        mol_arom = Chem.Mol(mol)

        edge_index = []
        edge_attr = []
        edge_type = []
        x = []
        smask_list = []
        wildcard_flags = []
        
        # Collect atom features from aromatic-aware molecule so that
        # atom.GetIsAromatic() remains available in features
        for atom in mol_arom.GetAtoms():
            atom_feats = atom_features(atom)
            x.append(atom_feats)
            smask_list.append(0 if atom.GetIdx() in smask else 1)
            # Wildcard detection: RDKit dummy atoms (atomic num 0) or symbol '*'
            is_wc = (atom.GetAtomicNum() == 0) or (atom.GetSymbol() == '*')
            wildcard_flags.append(1 if is_wc else 0)
            
        # Log the original feature dimension
        original_dim = len(x[0]) if x else 0
        logger.debug(f"Original atom feature dimension: {original_dim}")
        
        # Collect bond information from aromatic-aware molecule.
        # 3-edge mapping (0-based) for RGCNConv:
        # 0=SINGLE, 1=DOUBLE, 2=TRIPLE. Aromatic edges marked as -1 (excluded later).
        bond_type_map = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
        }
        for bond in mol_arom.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[u, v], [v, u]])
            bond_features = etype_features(bond)
            edge_attr.extend([bond_features, bond_features])
            # If aromatic, mark as -1 to exclude in model conv
            if bond.GetIsAromatic() or bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                bid = -1
            else:
                btype = bond.GetBondType()
                bid = bond_type_map.get(btype, 0)
            edge_type.extend([bid, bid])
        
        # Convert to tensors
        x = torch.tensor(np.array(x), dtype=torch.float)
        edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        edge_type = torch.tensor(np.array(edge_type), dtype=torch.long)
        # Sanity: ensure edge_type values are within [-1, 2]
        if edge_type.numel() > 0:
            mn = int(edge_type.min())
            mx = int(edge_type.max())
            if mn < -1 or mx > 2:
                logger.error(f"Edge type out of range [-1,2] for {smiles}: min={mn}, max={mx}")
                raise ValueError("edge_type indices must be within [-1,2] for 3-edge mapping with aromatic exclusion")
        smask_list = torch.tensor(np.array(smask_list), dtype=torch.float)
        is_wildcard = torch.tensor(np.array(wildcard_flags), dtype=torch.float)
        
        # Ensure x has exactly 41 features with detailed logging
        if x.shape[1] != 41:
            logger.warning(
                f"Node features dimension mismatch for SMILES {smiles}. "
                f"Current dimension: {x.shape[1]}, Required: 41"
            )
            
            if x.shape[1] > 41:
                logger.debug("Truncating features to first 41 dimensions")
                x = x[:, :41]
            else:
                logger.debug(f"Padding features with zeros (adding {41 - x.shape[1]} dimensions)")
                padding = torch.zeros(x.shape[0], 41 - x.shape[1])
                x = torch.cat([x, padding], dim=1)
            
            logger.info(f"Final feature dimension after adjustment: {x.shape[1]}")
        
        # Validate final dimensions
        assert x.shape[1] == 41, f"Feature dimension is {x.shape[1]}, expected 41"
        
        # Create and validate the graph data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
        data.smask = smask_list
        data.is_wildcard = is_wildcard
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        
        # Final validation
        #assert data.x.shape[1] == 41, f"Final feature dimension is {data.x.shape[1]}, expected 41"
        
        # Debug statement**
        #print(f"[construct_mol_graph_from_smiles] Graph features shape for SMILES {smiles}: {data.x.shape}")        
        
        return data
        
    except Exception as e:
        logger.error(f"Error constructing graph for SMILES {smiles}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

        
def _extract_label_row(labels: pd.DataFrame, idx: int):
    """Utility to extract a row of labels as list or value preserving NaNs."""
    if isinstance(labels, pd.DataFrame):
        return labels.iloc[idx].to_list()
    return labels.iloc[idx]


def build_mol_graph_data(dataset_smiles, labels_name, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)

    for i, smiles in enumerate(smilesList):
        try:
            canonical_smiles, _, _ = roundtrip_or_fallback(str(smiles))
            g_rgcn = construct_mol_graph_from_smiles(canonical_smiles, smask=[])
            label_row = _extract_label_row(labels, i)
            molecule = [canonical_smiles, g_rgcn, label_row, split_index.iloc[i]]
            dataset_gnn.append(molecule)
            logger.info(f'{i + 1}/{molecule_number} molecule is transformed to mol graph! {len(failed_molecule)} is transformed failed!')
        except Exception as e:
            logger.error(f'{smiles} is transformed to mol graph failed: {e}')
            failed_molecule.append(smiles)
    
    logger.info(f'{failed_molecule}({len(failed_molecule)}) is transformed to mol graph failed!')
    return dataset_gnn


def build_mol_graph_data_for_brics(dataset_smiles, labels_name, smiles_name):
    """
    Build molecular graphs with enhanced BRICS decomposition.
    """
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    
    for i, orig_smiles in enumerate(smilesList):
        try:
            substructure_dir = return_brics_leaf_structure(orig_smiles)
            
            # Create masks for both fragments and reaction centers
            atom_mask = []
            brics_substructure_mask = []
            reaction_center_mask = []
            
            # Process regular fragments
            for _, substructure in substructure_dir['substructure'].items():
                brics_substructure_mask.append(substructure)
                atom_mask = atom_mask + substructure
            
            # Process reaction centers
            for _, center in substructure_dir['reaction_centers'].items():
                reaction_center_mask.append(center)
            
            # Combine masks for processing
            smask = [[x] for x in range(len(atom_mask))] + brics_substructure_mask + reaction_center_mask
            
            # Create graphs for each mask
            for j, smask_i in enumerate(smask):
                try:
                    g = construct_mol_graph_from_smiles(orig_smiles, smask=smask_i)
                    
                    # Store additional information in the graph
                    g.brics_bond_types = substructure_dir['brics_bond_types']
                    g.is_reaction_center = j >= len(atom_mask) + len(brics_substructure_mask)

                    label_row = _extract_label_row(labels, i)
                    molecule = [orig_smiles, g, label_row, split_index.iloc[i], smask_i, i]
                    dataset_gnn.append(molecule)
                    
                    logger.info(f'{j + 1}/{len(smask)}, {i + 1}/{molecule_number} molecule processed')
                except Exception as e:
                    logger.error(f'{orig_smiles} with smask {smask_i} failed: {e}')
                    failed_molecule.append(orig_smiles)
        except Exception as e:
            logger.error(f'Error processing molecule {orig_smiles}: {e}')
            failed_molecule.append(orig_smiles)
    
    logger.info(f'{len(failed_molecule)} molecules failed processing')
    return dataset_gnn
    
def build_mol_graph_data_for_murcko(dataset_smiles, labels_name, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    
    for i, orig_smiles in enumerate(smilesList):
        try:
            substructure_dir = return_murcko_leaf_structure(orig_smiles)
            atom_mask = []
            murcko_substructure_mask = []
            for _, substructure in substructure_dir['substructure'].items():
                murcko_substructure_mask.append(substructure)
                atom_mask = atom_mask + substructure
            smask = murcko_substructure_mask
            
            for j, smask_i in enumerate(smask):
                try:
                    g = construct_mol_graph_from_smiles(orig_smiles, smask=smask_i)
                    label_row = _extract_label_row(labels, i)
                    molecule = [orig_smiles, g, label_row, split_index.iloc[i], smask_i, i]
                    dataset_gnn.append(molecule)
                    logger.info(f'{j + 1}/{len(smask)}, {i + 1}/{molecule_number} molecule is transformed to mol graph! {len(failed_molecule)} is transformed failed!')
                except Exception as e:
                    logger.error(f'{orig_smiles} with smask {smask_i} is transformed to mol graph failed: {e}')
                    failed_molecule.append(orig_smiles)
        except Exception as e:
            logger.error(f'Error processing molecule {orig_smiles}: {e}')
            failed_molecule.append(orig_smiles)
    
    logger.info(f'{failed_molecule}({len(failed_molecule)}) is transformed to mol graph failed!')
    return dataset_gnn



def build_mol_graph_data_for_fg(dataset_smiles, labels_name, smiles_name, config: Configuration):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    
    # Get SMARTS patterns from config
    fg_with_ca_smarts = config.fg_with_ca_smart
    fg_without_ca_smarts = config.fg_without_ca_smart
    fg_name_list = [f'fg_{i}' for i in range(len(fg_with_ca_smarts))]
    
    # Precompile SMARTS patterns
    fg_with_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_with_ca_smarts]
    fg_without_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_without_ca_smarts]
    
    for i, orig_smiles in enumerate(smilesList):
        try:
            hit_at_wash, hit_fg_name_wash = return_fg_hit_atom(orig_smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list)
            atom_mask = []  # Added to match BRICS/Murcko pattern
            fg_substructure_mask = []  # Added to match BRICS/Murcko pattern
            
            # Construct mask list similar to BRICS/Murcko
            for fg_atoms in hit_at_wash:
                fg_substructure_mask.append(fg_atoms)
                atom_mask = atom_mask + fg_atoms
            smask = fg_substructure_mask  # Match BRICS/Murcko pattern
            
            for j, smask_i in enumerate(smask):
                try:
                    g = construct_mol_graph_from_smiles(orig_smiles, smask=smask_i)
                    label_row = _extract_label_row(labels, i)
                    molecule = [orig_smiles, g, label_row, split_index.iloc[i], smask_i, i]
                    dataset_gnn.append(molecule)
                    logger.info(f'{j + 1}/{len(smask)}, {i + 1}/{molecule_number} molecule is transformed to mol graph! {len(failed_molecule)} is transformed failed!')
                except Exception as e:
                    logger.error(f'{orig_smiles} with smask {smask_i} is transformed to mol graph failed: {e}')
                    failed_molecule.append(orig_smiles)
        except Exception as e:
            logger.error(f'Error processing molecule {orig_smiles}: {e}')
            failed_molecule.append(orig_smiles)
    
    logger.info(f'{failed_molecule}({len(failed_molecule)}) is transformed to mol graph failed!')
    return dataset_gnn

def save_dataset(
    dataset: Any,
    sub_type: str,
    config: Configuration,
    graph_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    smask_save_path: Optional[str] = None
) -> None:
    """
    Saves the dataset and its corresponding metadata while maintaining compatibility 
    with the old output format.

    Args:
        dataset (Any): The dataset object to save.
        sub_type (str): The substructure type ('primary', 'brics', 'murcko', 'fg').
        config (Configuration): Configuration object containing paths and task details.
        graph_path (Optional[str]): Custom path to save the graph data.
        meta_path (Optional[str]): Custom path to save the meta data.
        smask_save_path (Optional[str]): Custom path to save the smask data (for non-primary types).
    """
    if not dataset:
        logger.info(f"No {sub_type} molecules were successfully processed.")
        return

    # Determine task type based on configuration
    task_type = 'classification' if config.classification else 'regression'

    label_cols = config.labels_name if isinstance(config.labels_name, list) else [config.labels_name]

    if sub_type == 'primary':
        smiles, g_pyg, labels, split_index = map(list, zip(*dataset))
        graph_labels = {'labels': torch.tensor(labels)}

        split_index_pd = pd.DataFrame({'smiles': smiles, 'group': split_index})
        split_index_pd[config.compound_id_name] = config.dataset_origin[config.compound_id_name]
        for col in label_cols:
            split_index_pd[col] = config.dataset_origin[col]

        # Compute representation summary and add columns
        repr_used_records = []
        used_flags = []
        text_reprs = []
        for s in smiles:
            can_smi, text_repr, used_big = roundtrip_or_fallback(s)
            text_reprs.append(text_repr)
            used_flags.append(bool(used_big))
            repr_used_records.append({'smiles': can_smi, 'text_repr': text_repr, 'used_bigsmiles': bool(used_big)})

        split_index_pd['text_repr'] = text_reprs
        split_index_pd['used_bigsmiles'] = used_flags

        # Save representation usage summary CSV and log coverage
        repr_df = pd.DataFrame(repr_used_records)
        repr_csv_path = os.path.join(config.output_dir, 'repr_used.csv')
        try:
            repr_df.to_csv(repr_csv_path, index=False)
        except Exception:
            # Fallback to workspace root if output dir unwritable
            repr_df.to_csv('repr_used.csv', index=False)
            repr_csv_path = os.path.abspath('repr_used.csv')
        coverage = 100.0 * (sum(used_flags) / max(1, len(used_flags)))
        logger.info(f'BigSMILES coverage: {coverage:.1f}% (saved {repr_csv_path})')

        # Save files with custom or default paths
        if graph_path is None:
            graph_filename = f"{config.task_name}_{task_type}_{sub_type}_graphs.pt"
            graph_path = os.path.join(config.output_dir, graph_filename)
        if meta_path is None:
            meta_filename = f"{config.task_name}_{task_type}_{sub_type}_meta.csv"
            meta_path = os.path.join(config.output_dir, meta_filename)
        
        split_index_pd.to_csv(meta_path, index=False)
        torch.save((g_pyg, graph_labels), graph_path)
        
    else:
        smiles, g_pyg, labels, split_index, smask, orig_indices = map(list, zip(*dataset))
        graph_labels = {'labels': torch.tensor(labels)}

        split_index_pd = pd.DataFrame({'smiles': smiles, 'group': split_index})
        split_index_pd[config.compound_id_name] = [config.dataset_origin[config.compound_id_name].iloc[i] for i in orig_indices]
        for col in label_cols:
            split_index_pd[col] = [config.dataset_origin[col].iloc[i] for i in orig_indices]

        # Save files with custom or default paths
        if graph_path is None:
            graph_filename = f"{config.task_name}_{task_type}_{sub_type}_graphs.pt"
            graph_path = os.path.join(config.output_dir, graph_filename)
        if meta_path is None:
            meta_filename = f"{config.task_name}_{task_type}_{sub_type}_meta.csv"
            meta_path = os.path.join(config.output_dir, meta_filename)
        if smask_save_path is None:
            smask_filename = f"{config.task_name}_{task_type}_{sub_type}_smask.npy"
            smask_save_path = os.path.join(config.output_dir, smask_filename)
        
        split_index_pd.to_csv(meta_path, index=False)
        torch.save((g_pyg, graph_labels), graph_path)
        np.save(smask_save_path, np.array(smask, dtype=object))

    logger.info(f'{sub_type.capitalize()} molecules graph and metadata saved successfully!')

def main():
    # Initialize Configuration
    config = Configuration()
    config.validate()  # Ensure configurations are valid

    # Set random seed for reproducibility
    config.set_seed(seed=42)  # You can choose any seed value

    # Load your dataset and set it in config
    dataset_smiles = pd.read_csv(config.origin_data_path)
    config.dataset_origin = dataset_smiles  # Add this line
    
    # Build only primary dataset (3-edge relations: SINGLE/DOUBLE/TRIPLE)
    logger.info("Building primary molecule graphs (3-edge v2: single/double/triple; aromatic marked -1 and excluded in model)...")
    dataset_gnn_primary = build_mol_graph_data(dataset_smiles, config.labels_name, config.smiles_name)

    if dataset_gnn_primary:
        logger.info("Saving primary molecule graphs and metadata (3-edge v2)...")
        save_dataset(dataset_gnn_primary, 'primary', config)
    else:
        logger.warning("No primary molecules were processed successfully.")

    # Skip BRICS, Murcko, and FG graphs for 3-edge build
    logger.info("Skipping BRICS/Murcko/FG graph generation for 3-edge pipeline.")

if __name__ == "__main__":
    main()
