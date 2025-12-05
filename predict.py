import os
import re
import torch
import numpy as np
import pandas as pd
import argparse
import logging
import traceback
import signal
from datetime import datetime
import time
import pickle
import math
from typing import Dict, Any, List, Tuple, Optional
from torch_geometric.loader import DataLoader as PyGDataLoader
from collections import deque
import itertools
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, SanitizeFlags, SanitizeMol
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
import concurrent.futures
import gc
import traceback
import itertools
from collections import deque
from rdkit.Chem import MolFragmentToSmiles
from torch_geometric.data import Batch, Data
import concurrent.futures
import sys
import json
from model import BaseGNN
from bigsmiles_utils import roundtrip_or_fallback
from data_module import MoleculeDataset
from build_data import (
    return_murcko_leaf_structure,
    return_brics_leaf_structure,
    return_fg_hit_atom,
    construct_mol_graph_from_smiles,
)
from logger import get_logger
from config import Configuration
try:
    import joblib
except Exception:
    joblib = None
from pathlib import Path
try:
    from xai_agreement import add_agreement_to_xai_json
except ImportError:
    add_agreement_to_xai_json = None
# Global property names and competition weights for polymer regression
PROPERTY_NAMES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
STRATEGIC_WEIGHTS = {
    'Tg': 1.0,
    'FFV': 0.073,
    'Tc': 0.693,
    'Density': 0.834,
    'Rg': 0.832,
}


def calculate_strategic_wmae(predictions: Dict[str, float]) -> float:
    """Estimate weighted MAE from predictions only."""
    comps = []
    for prop, weight in STRATEGIC_WEIGHTS.items():
        key = f"pred_{prop}"
        if key in predictions:
            comps.append(weight * abs(predictions[key]))
    return sum(comps) / sum(STRATEGIC_WEIGHTS.values()) if comps else 0.0




# Silence RDKit kekulize warnings
RDLogger.DisableLog('rdApp.error')
def atom_disjoint_attribution(mol: Chem.Mol, fragment_atoms_list: List[List[int]],
                              fragment_attributions: List[float]) -> Tuple[np.ndarray, List[float]]:
    """Aggregate fragment attributions at the atom level and recompute per-fragment scores."""
    n_atoms = mol.GetNumAtoms()
    atom_sums = np.zeros(n_atoms)
    atom_counts = np.zeros(n_atoms)

    for frag_atoms, frag_attr in zip(fragment_atoms_list, fragment_attributions):
        for idx in frag_atoms:
            if idx < n_atoms:
                atom_sums[idx] += frag_attr
                atom_counts[idx] += 1

    atom_counts[atom_counts == 0] = 1
    atom_avg = atom_sums / atom_counts

    corrected = [float(np.sum(atom_avg[frag])) for frag in fragment_atoms_list]

    return atom_avg, corrected


RDLogger.DisableLog('rdApp.warning')

def smiles_to_mol_no_kekule(smi):
    # build without sanitization
    m = MolFromSmiles(smi, sanitize=False)
    # sanitize everything except Kekulize
    SanitizeMol(m, 
        SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE
    )
    return m



logger = get_logger(__name__)

# Minimal GPU memory guard; small scope only
class GPUMemoryGuard:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        try:
            gc.collect()
        except Exception:
            pass
class PredictionManager:
    def __init__(self, config: Configuration, input_csv: str, device: Optional[str] = None, num_workers: int = 4, ensemble_type: str = 'best5'):
        """Initialize PredictionManager with unique checkpoint file per input file."""
        # Initialize logger first
        self.logger = get_logger(__name__)
        self.logger.info("Initializing PredictionManager...")
        
        self.ensemble_type = ensemble_type
        # Store configuration and parameters
        self.config = config
        self.num_workers = num_workers
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Generate unique checkpoint filename based on input file and current timestamp
        input_basename = os.path.splitext(os.path.basename(input_csv))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = os.path.join(os.path.dirname(input_csv), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create unique checkpoint filename using input file name and timestamp
        self.checkpoint_file = os.path.join(
            checkpoint_dir,
            f"prediction_checkpoint_{input_basename}.pkl"
        )
        self.logger.info(f"Using checkpoint file: {self.checkpoint_file}")
        
        # Initialize other attributes
        self.models = []
        self.should_terminate = False
        self.input_csv = input_csv  # Store input file path for reference
        
        # Initialize process pool for parallel processing
        if torch.cuda.is_available():
            torch.multiprocessing.set_start_method('spawn', force=True)
            
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize functional group patterns if needed
        if hasattr(self.config, 'fg_with_ca_smart'):
            self._initialize_fg_patterns()
        
        # Set prediction mode
        self.config.prediction_mode = True

        # Representation selection controls for inference
        # Allow fallback to config values; CLI may override them before constructing PredictionManager
        if not hasattr(self.config, 'PREDICT_REPR'):
            self.config.PREDICT_REPR = 'auto'
        if not hasattr(self.config, 'BIGSMILES_REQUIRED'):
            self.config.BIGSMILES_REQUIRED = False
        if not hasattr(self.config, 'USE_BIGSMILES'):
            self.config.USE_BIGSMILES = True
        # BigSMILES coverage counters
        self._bigs_ok = 0
        self._bigs_total = 0
        # Map compound_id -> {canon_smiles, text_repr, used_bigsmiles}
        self._repr_map = {}
        
        # Load models
        self._load_models()

        # Load optional FFV calibrator
        self.ffv_calibrator = self._load_ffv_calibrator(
            getattr(self.config, 'CALIBRATION_DIR', getattr(self.config, 'CALIBRATION_OUTDIR', 'calibration/')),
            enabled=getattr(self.config, 'CALIBRATE_FFV', False)
        )

        # Ensure config defaults present for XAI options
        if not hasattr(self.config, 'XAI_TOPK'):
            self.config.XAI_TOPK = 10
        if not hasattr(self.config, 'EMIT_XAI_JSON'):
            self.config.EMIT_XAI_JSON = True
        # Prepare XAI output directories
        self.xai_outdir = os.path.join(
            getattr(self.config, 'output_dir', os.path.dirname(input_csv)),
            getattr(self.config, 'XAI_JSON_OUTDIR', 'xai/').strip('/')
        )
        os.makedirs(self.xai_outdir, exist_ok=True)
        self.xai_index_path = os.path.join(self.xai_outdir, '_index.jsonl')

        # Load or estimate training-set mean FFV for residual-aware reporting
        self.mu_ffv = self._load_mu_ffv()

        # Initialize adaptive gates from OOF if needed
        self._init_explainability_controller()

    # ---------------------
    # Explainability Controller
    # ---------------------
    def _init_explainability_controller(self):
        # Buffer tracking recent heavy-XAI decisions
        from collections import deque as _dq
        self._heavy_window = _dq(maxlen=int(getattr(self.config, 'EXPL_RATE_WINDOW', 200)))
        # Initialize thresholds (tau_std, delta_min)
        tau = getattr(self.config, 'TAU_STD', None)
        dmin = getattr(self.config, 'DELTA_MIN', None)
        # Read OOF MAE from stack_report.json if available
        mae_oof = None
        try:
            rep = os.path.join(self.config.output_dir, f"{self.config.task_name}_{self.config.task_type}_final_eval", 'stack_report.json')
            if os.path.exists(rep):
                with open(rep, 'r') as f:
                    js = json.load(f)
                    ffv = js.get('FFV', {})
                    mae_oof = float(ffv.get('stack_oof_mae', ffv.get('base_oof_mae', float('nan'))))
        except Exception:
            mae_oof = None
        # Estimate dynamic range from property ranges fallback
        pr = getattr(self.config, 'property_ranges', {}).get('FFV', (0.2, 0.8))
        dr = float(abs(pr[1] - pr[0])) if isinstance(pr, (list, tuple)) and len(pr) == 2 else 0.5
        if tau is None or dmin is None:
            # Defaults if no OOF available
            if mae_oof is None or not np.isfinite(mae_oof):
                mae_oof = 0.02
            tau0 = 0.4 * mae_oof
            dmin0 = 0.6 * mae_oof
            # Floors as fraction of dynamic range
            tau0 = max(tau0, 0.02 * dr)
            dmin0 = max(dmin0, 0.05 * dr)
            if tau is None:
                self.config.TAU_STD = tau0
            if dmin is None:
                self.config.DELTA_MIN = dmin0
        self.logger.info(f"Explainability gates initialized: TAU_STD={self.config.TAU_STD:.4f}, DELTA_MIN={self.config.DELTA_MIN:.4f}")

    
    def _controller_update(self, batch_heavy_flags: List[bool], avg_secs_per_mol: Optional[float] = None):
        try:
            for b in batch_heavy_flags:
                self._heavy_window.append(bool(b))

            n_win = len(self._heavy_window)
            min_win = max(50, int(getattr(self.config, 'EXPL_RATE_WINDOW', 200) * 0.5))
            if n_win < min_win:
                return  # warm-up: not enough evidence yet

            target = float(getattr(self.config, 'DESIRED_EXPL_RATE', 0.5))
            tol    = float(getattr(self.config, 'EXPL_RATE_TOL', 0.20))
            rate   = float(sum(self._heavy_window)) / float(n_win)

            relax  = getattr(self.config, 'ADJ_RELAX', (1.25, 0.90))
            tight  = getattr(self.config, 'ADJ_TIGHT', (0.90, 1.10))

            tau_old  = float(getattr(self.config, 'TAU_STD', 0.02))
            dmin_old = float(getattr(self.config, 'DELTA_MIN', 0.01))
            tau, dmin = tau_old, dmin_old

            budget   = float(getattr(self.config, 'XAI_BUDGET_S_PER_MOL', 8.0))
            too_slow = (avg_secs_per_mol is not None) and (avg_secs_per_mol > budget)

            changed = False
            if rate < target * (1 - tol) and not too_slow:
                tau  *= float(relax[0]); dmin *= float(relax[1]); changed = True
            elif rate > target * (1 + tol) or too_slow:
                tau  *= float(tight[0]); dmin *= float(tight[1]); changed = True

            # Clamp
            tau  = max(0.001, min(tau, 0.05))
            dmin = max(0.002, min(dmin, 0.05))

            if changed and (tau != tau_old or dmin != dmin_old):
                self.config.TAU_STD  = tau
                self.config.DELTA_MIN = dmin
                self.logger.info(
                    f"XAI controller: rate={rate:.2f} target={target:.2f} "
                    f"tau={tau_old:.3f}->{tau:.3f} delta={dmin_old:.3f}->{dmin:.3f} "
                    f"avg={avg_secs_per_mol or 0:.1f}s")
        except Exception as e:
            self.logger.debug(f"ExplainabilityController update skipped: {e}")


    def _light_xai(self, smiles: str, K: int) -> List[Dict[str, Any]]:
        """Fast, no-mask explanation using atom weights aggregated to top-K polymer masks."""
        try:
            # One forward on first model to get atom saliency/weights
            if not self.models:
                return []
            graph = self._get_graph(smiles, mask=[])
            if graph is None:
                return []
            preds, atom_w = self._predict_single_model(self.models[0], graph)
            if atom_w is None:
                return []
            mol = self._get_mol(smiles)
            if mol is None:
                return []
            # Build polymer-masks and pick disjoint K
            cands = []
            for atoms in self.build_polymer_masks(mol):
                if atoms:
                    cands.append({'mask_id': 'POLY', 'atoms': list(map(int, atoms)), 'mask_family': 'fragment'})
            selected = self._select_non_overlapping_masks(cands, K)
            # Aggregate atom weights
            frags = []
            for m in selected:
                idxs = m['atoms']
                score = float(np.mean([atom_w[i] if 0 <= i < len(atom_w) else 0.0 for i in idxs]))
                frags.append({
                    'mask_id': m['mask_id'],
                    'mask_family': m.get('mask_family', 'fragment'),
                    'atoms': idxs,
                    'attr': score
                })
            # Return top +/- split
            pos = sorted([f for f in frags if f['attr'] >= 0], key=lambda x: abs(x['attr']), reverse=True)
            neg = sorted([f for f in frags if f['attr'] < 0], key=lambda x: abs(x['attr']), reverse=True)
            k_pos = K // 2
            k_neg = K - k_pos
            return pos[:k_pos] + neg[:k_neg]
        except Exception:
            return []

    # ---------------------
    # Representation selection helper
    # ---------------------
    def _choose_text_repr(self, smiles: str, mode: str = "auto", require_bigs: bool = False) -> tuple[str, str, bool]:
        """Return (canon_smiles, text_repr, used_bigsmiles) using round-trip logic.
        mode: 'auto'|'smiles'|'bigsmiles'
        """
        # Ensure string
        s = str(smiles) if smiles is not None else ''
        try:
            if mode == 'smiles':
                # canonicalize via roundtrip helper to be consistent (it canonicalizes internally)
                sm2, _, _ = roundtrip_or_fallback(s)
                return sm2, sm2, False
            if mode == 'bigsmiles':
                sm2, bs, ok = roundtrip_or_fallback(s)
                if not ok and require_bigs:
                    raise ValueError('BigSMILES required but conversion failed')
                return sm2, (bs if ok else sm2), bool(ok)
            # auto
            sm2, bs, ok = roundtrip_or_fallback(s)
            if ok:
                return sm2, bs, True
            if require_bigs:
                raise ValueError('BigSMILES required but conversion failed')
            return sm2, sm2, False
        except Exception:
            if require_bigs:
                raise
            # fallback to SMILES
            return s, s, False

    # ---------------------
    # Mask collection and Δ-masking helpers
    # ---------------------
    def _collect_candidate_masks(self, smiles: str) -> List[Dict[str, Any]]:
        """
        Build candidate mask sets from BRICS/Murcko/FG (fragment family) and, if available,
        BigSMILES blocks/connectors/end-groups (bigsmiles family types).
        De-duplicates masks by atom set.
        """
        masks: List[Dict[str, Any]] = []
        seen = set()

        # Fragments from Murcko
        try:
            murcko = self.get_substructures(smiles, 'murcko') or {}
            for idx, atoms in murcko.items():
                a = tuple(sorted(map(int, atoms)))
                if a in seen:
                    continue
                seen.add(a)
                masks.append({
                    'mask_id': f'MURCKO_{int(idx):02d}',
                    'atoms': list(a),
                    'mask_family': 'fragment',
                    'source': 'murcko'
                })
        except Exception as e:
            self.logger.debug(f"Murcko mask collection failed: {e}")

        # Fragments from BRICS
        try:
            brics = self.get_substructures(smiles, 'brics') or {}
            for idx, atoms in brics.items():
                a = tuple(sorted(map(int, atoms)))
                if a in seen:
                    continue
                seen.add(a)
                masks.append({
                    'mask_id': f'BRICS_{int(idx):02d}',
                    'atoms': list(a),
                    'mask_family': 'fragment',
                    'source': 'brics'
                })
        except Exception as e:
            self.logger.debug(f"BRICS mask collection failed: {e}")

        # Fragments from FG
        try:
            fg = self.get_substructures(smiles, 'fg') or {}
            for idx, atoms in fg.items():
                a = tuple(sorted(map(int, atoms)))
                if a in seen:
                    continue
                seen.add(a)
                masks.append({
                    'mask_id': f'FG_{int(idx):02d}',
                    'atoms': list(a),
                    'mask_family': 'fragment',
                    'source': 'fg'
                })
        except Exception as e:
            self.logger.debug(f"FG mask collection failed: {e}")

        # BigSMILES families (block/connector/end_group) - mapping not available in this pass
        # We will add a note in the JSON if unavailable; keep this section as a placeholder
        # Polymer-specific linear fallbacks: for acyclic RU, add backbone windows and linkage motifs
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and not mol.GetRingInfo().AtomRings():
                extra = self._linear_polymer_masks(mol)
                # de-duplicate against seen
                for m in extra:
                    a = tuple(sorted(map(int, m.get('atoms', []))))
                    if a in seen or not a:
                        continue
                    seen.add(a)
                    masks.append(m)
        except Exception as e:
            self.logger.debug(f"Linear polymer mask fallback failed: {e}")

        return masks

    def _linear_polymer_masks(self, mol: Chem.Mol) -> List[Dict[str, Any]]:
        """Add masks suitable for ring-free (linear) repeating units.
        - Linkage motifs (ester/amide/ether/urethane/sulfone) as small masks
        - Backbone sliding windows along approximate diameter path
        """
        masks: List[Dict[str, Any]] = []

        # 1) Linkage motifs (expand by 1 bond to capture local context)
        motif_smarts = {
            'ester': 'C(=O)O',
            'amide': 'C(=O)N',
            'urethane': 'OC(=O)N',
            'carbonate': 'OC(=O)O',
            'ether': 'C-O-C',
            'sulfone': 'S(=O)(=O)'
        }
        for name, patt in motif_smarts.items():
            try:
                q = Chem.MolFromSmarts(patt)
                if not q:
                    continue
                for match in mol.GetSubstructMatches(q):
                    atoms = set(int(i) for i in match)
                    # expand by 1 bond
                    for i in list(atoms):
                        for n in mol.GetAtomWithIdx(i).GetNeighbors():
                            atoms.add(int(n.GetIdx()))
                    masks.append({
                        'mask_id': f'MOTIF_{name.upper()}_{len(masks):02d}',
                        'atoms': sorted(list(atoms)),
                        'mask_family': 'fragment',
                        'source': f'motif:{name}'
                    })
            except Exception:
                continue

        # 2) Backbone sliding windows along approximate diameter path
        try:
            path = self._approx_diameter_path(mol)
            if path:
                k = 6  # window length in atoms
                stride = 3
                max_windows = 8
                windows = []
                for start in range(0, max(1, len(path) - k + 1), stride):
                    win_atoms = path[start:start + k]
                    if not win_atoms:
                        continue
                    windows.append(list(map(int, win_atoms)))
                    if len(windows) >= max_windows:
                        break
                for i, at in enumerate(windows):
                    masks.append({
                        'mask_id': f'BACKBONE_{i:02d}',
                        'atoms': sorted(at),
                        'mask_family': 'fragment',
                        'source': 'backbone_window'
                    })
        except Exception:
            pass

        return masks

    def _get_mol(self, smiles: str) -> Optional[Chem.Mol]:
        if not hasattr(self, '_mol_cache'):
            self._mol_cache = {}
        m = self._mol_cache.get(smiles)
        if m is not None:
            return m
        try:
            m = Chem.MolFromSmiles(smiles)
            if m is not None:
                if len(self._mol_cache) > 128:
                    self._mol_cache.clear()
                self._mol_cache[smiles] = m
            return m
        except Exception:
            return None

    def _get_graph(self, smiles: str, mask: Optional[List[int]] = None):
        if not hasattr(self, '_graph_cache'):
            self._graph_cache = {}
        key = (smiles, tuple(sorted(mask)) if mask else ())
        g = self._graph_cache.get(key)
        if g is not None:
            return g
        g = construct_mol_graph_from_smiles(smiles, smask=(mask or []))  # base graph; _get_graph wrapper used elsewhere
        if g is not None:
            if len(self._graph_cache) > 64:
                # simple LRU-like clear
                self._graph_cache.clear()
            self._graph_cache[key] = g
        return g

    def build_polymer_masks(self, mol: Chem.Mol) -> List[List[int]]:
        """Heuristic polymer-aware Murcko masks.
        Returns a list of atom-index lists (candidate masks) before de-overlap selection.
        """
        masks: List[List[int]] = []
        if mol is None:
            return masks
        n = mol.GetNumAtoms()
        if n == 0:
            return masks
        # Backbone: between wildcard '*' atoms if present, else diameter path
        star_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(star_idxs) >= 2:
            # pick farthest pair among stars using BFS distances
            def bfs_dist(src: int) -> Dict[int, int]:
                from collections import deque as _dq
                q = _dq([src])
                dist = {src: 0}
                while q:
                    u = q.popleft()
                    for v in mol.GetAtomWithIdx(u).GetNeighbors():
                        vi = v.GetIdx()
                        if vi not in dist:
                            dist[vi] = dist[u] + 1
                            q.append(vi)
                return dist
            best_pair = (star_idxs[0], star_idxs[1])
            best_d = -1
            for i in range(len(star_idxs)):
                dmap = bfs_dist(star_idxs[i])
                for j in range(i + 1, len(star_idxs)):
                    dj = dmap.get(star_idxs[j], -1)
                    if dj > best_d:
                        best_d = dj
                        best_pair = (star_idxs[i], star_idxs[j])
            # reconstruct path via parent tracking
            # second BFS to get parent
            src, dst = best_pair
            from collections import deque as _dq
            q = _dq([src])
            parent = {src: -1}
            seen = {src}
            while q:
                u = q.popleft()
                if u == dst:
                    break
                for v in mol.GetAtomWithIdx(u).GetNeighbors():
                    vi = v.GetIdx()
                    if vi not in seen:
                        seen.add(vi)
                        parent[vi] = u
                        q.append(vi)
            path = [dst]
            cur = dst
            while parent.get(cur, -1) != -1:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            if path:
                masks.append(list(map(int, path)))
        else:
            path = self._approx_diameter_path(mol)
            if path:
                masks.append(list(map(int, path)))
        # Ring systems (fused groups)
        ring_sets = [set(r) for r in mol.GetRingInfo().AtomRings()]
        # Union-find fused rings
        parent = list(range(len(ring_sets)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for i in range(len(ring_sets)):
            for j in range(i+1, len(ring_sets)):
                if ring_sets[i] & ring_sets[j]:
                    union(i,j)
        fused_groups: Dict[int, set] = {}
        for i, s in enumerate(ring_sets):
            r = find(i)
            fused_groups.setdefault(r, set()).update(s)
        fused_list = sorted(fused_groups.values(), key=lambda s: -len(s))[:2]
        for i, g in enumerate(fused_list):
            if g:
                masks.append(sorted(list(g)))
        # Linkers (non-ring atoms connecting ring groups)
        ring_atoms_all = set().union(*ring_sets) if ring_sets else set()
        linkers: List[List[int]] = []
        for a in range(n):
            if a in ring_atoms_all:
                continue
            atom = mol.GetAtomWithIdx(a)
            ring_nbrs = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetIdx() in ring_atoms_all]
            if len(ring_nbrs) >= 2:
                # small neighborhood mask around linker
                local = set([a])
                for nb in atom.GetNeighbors():
                    local.add(nb.GetIdx())
                    for nb2 in nb.GetNeighbors():
                        if nb2.GetIdx() != a:
                            local.add(nb2.GetIdx())
                linkers.append(sorted(list(local)))
        # de-dup and take top2 largest
        uniq = []
        for l in linkers:
            s = set(l)
            if not any(s == set(u) for u in uniq):
                uniq.append(l)
        linkers = sorted(uniq, key=lambda v: -len(v))[:2]
        masks.extend(linkers)
        # End-groups near '*' atoms
        for si in star_idxs[:2]:
            local = set([si])
            for nb in mol.GetAtomWithIdx(si).GetNeighbors():
                local.add(nb.GetIdx())
                for nb2 in nb.GetNeighbors():
                    local.add(nb2.GetIdx())
            masks.append(sorted(list(local)))
        # Side-chains relative to backbone (first mask)
        if masks:
            backbone = set(masks[0])
            all_idx = set(range(n))
            side = list(all_idx - backbone)
            # BFS components in side
            nbrs = {i: [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors()] for i in range(n)}
            seen = set()
            comps = []
            for s in side:
                if s in seen:
                    continue
                if s in backbone:
                    continue
                comp = set()
                stack = [s]
                while stack:
                    u = stack.pop()
                    if u in seen or u in backbone or u not in side:
                        continue
                    seen.add(u)
                    comp.add(u)
                    for v in nbrs[u]:
                        if v not in seen:
                            stack.append(v)
                if comp:
                    comps.append(sorted(list(comp)))
            comps = sorted(comps, key=lambda v: -len(v))[:2]
            masks.extend(comps)
        return masks

    def _approx_diameter_path(self, mol: Chem.Mol) -> List[int]:
        """Approximate longest path by two BFS passes from a terminal atom.
        Returns list of atom indices forming a path.
        """
        g = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()] for a in mol.GetAtoms()}

        # pick a terminal heavy atom if available
        terminals = [i for i, nbrs in g.items() if len(nbrs) == 1]
        start = terminals[0] if terminals else 0

        def bfs_far(src: int) -> Tuple[int, Dict[int, int]]:
            from collections import deque
            q = deque([src])
            seen = {src}
            parent = {src: -1}
            last = src
            while q:
                u = q.popleft()
                last = u
                for v in g[u]:
                    if v not in seen:
                        seen.add(v)
                        parent[v] = u
                        q.append(v)
            return last, parent

        u, _ = bfs_far(start)
        v, parent = bfs_far(u)
        # reconstruct u->v path
        path = [v]
        cur = v
        while parent.get(cur, -1) != -1:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    def _select_non_overlapping_masks(self, masks: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
        """
        Greedy selection to avoid overlap. Priority:
          block, connector, end_group, fragment; then size desc; then mask_id.
        Returns up to topk masks with disjoint atom sets.
        """
        family_rank = {'block': 0, 'connector': 1, 'end_group': 2, 'fragment': 3}
        def key(m):
            return (family_rank.get(m.get('mask_family', 'fragment'), 9), -len(m.get('atoms', [])), m.get('mask_id', ''))

        sorted_masks = sorted(masks, key=key)
        used_atoms: set = set()
        selected: List[Dict[str, Any]] = []
        for m in sorted_masks:
            a = set(m.get('atoms', []))
            if not a:
                continue
            if a.isdisjoint(used_atoms):
                selected.append(m)
                used_atoms |= a
                if len(selected) >= topk:
                    break
        return selected

    def _compute_mask_deltas(self, smiles: str, mask_list: List[Dict[str, Any]], target_name: str) -> List[Dict[str, Any]]:
        """
        Compute raw and normalized deltas per mask using ensemble mean predictions.
        """
        # Original predictions per model
        per_model_orig: List[float] = []
        for model_info in self.models:
            preds = self._predict_single_model_regression(model_info, smiles)
            if preds is None or target_name not in preds:
                continue
            per_model_orig.append(float(preds[target_name]))
        if not per_model_orig:
            return []
        orig_mean = float(np.mean(per_model_orig))

        augmented: List[Dict[str, Any]] = []
        for m in mask_list:
            per_model_masked: List[float] = []
            for model_info in self.models:
                preds_m = self._predict_single_model_regression_masked(model_info, smiles, m['atoms'])
                if preds_m is None or target_name not in preds_m:
                    continue
                per_model_masked.append(float(preds_m[target_name]))
            if not per_model_masked:
                continue
            masked_mean = float(np.mean(per_model_masked))
            raw_delta = float(orig_mean - masked_mean)
            size = int(len(m['atoms']))
            norm_delta = float(raw_delta / max(1.0, size ** 0.5))
            mm = dict(m)
            mm.update({'raw_delta': raw_delta, 'norm_delta': norm_delta, 'size': size})
            augmented.append(mm)
        return augmented
        
    def _initialize_fg_patterns(self):
        """Initialize functional group patterns."""
        if not hasattr(self.config, 'fg_with_ca_list'):
            self.config.fg_name_list = [f'fg_{i}' for i in range(len(self.config.fg_with_ca_smart))]
            self.config.fg_with_ca_list = [Chem.MolFromSmarts(s) for s in self.config.fg_with_ca_smart]
            self.config.fg_without_ca_list = [Chem.MolFromSmarts(s) for s in self.config.fg_without_ca_smart]

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        self.logger.info("Termination signal received. Will exit after current batch.")
        self.should_terminate = True



    def _load_models(self):
        """Load models based on ensemble type."""
        if self.ensemble_type == 'full':
            return self._load_full_ensemble()
        else:
            return self._load_best_five()

    def _load_full_ensemble(self):
        """Load all models from CV runs."""
        cv_repeats = self.config.statistical_validation['cv_repeats']
        cv_folds = self.config.statistical_validation['cv_folds']
        
        self.logger.info(f"\nLoading all models from CV runs...")
        
        for cv_num in range(1, cv_repeats + 1):
            for fold in range(1, cv_folds + 1):
                try:
                    self._load_single_model(cv_num, fold)
                except Exception as e:
                    self.logger.error(f"Error loading model for CV{cv_num}, Fold{fold}: {str(e)}")
                    continue




    def _load_best_five(self):
        """Load best 5 models with intelligent checkpoint path resolution."""
        # Try primary location (directly in final_eval directory)
        best_models_file = os.path.join(
            self.config.output_dir,
            f"{self.config.task_name}_{self.config.task_type}_final_eval",
            "best_models.json",
        )

        # Fallback to legacy location (best_models subdirectory)
        if not os.path.exists(best_models_file):
            best_models_file = os.path.join(
                self.config.output_dir,
                f"{self.config.task_name}_{self.config.task_type}_final_eval",
                "best_models",
                "best_models.json",
            )

        with open(best_models_file, "r") as f:
            best_models_data = json.load(f)

        for model_info in best_models_data.get("models", []):
            cv_num = model_info["cv"]
            fold = model_info["fold"]

            checkpoint_path = os.path.join(
                self.config.output_dir,
                f"{self.config.task_name}_{self.config.task_type}_cv_results",
                f"cv{cv_num}",
                "checkpoints",
                f"{self.config.task_name}_{self.config.task_type}_cv{cv_num}_fold{fold}_best.ckpt",
            )

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

            self._load_single_model(cv_num, fold, checkpoint_path)


    def _load_single_model(self, cv_num: int, fold: int, checkpoint_path: str = None):
        """Load a single model with proper error handling for regression."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.config.output_dir,
                f"{self.config.task_name}_{self.config.task_type}_cv_results",
                f"cv{cv_num}",
                "checkpoints",
                f"{self.config.task_name}_{self.config.task_type}_cv{cv_num}_fold{fold}_best.ckpt",
            )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        hyperparams = checkpoint['hyperparameters']
        
        # Force regression mode regardless of what's in hyperparams
        model = BaseGNN(
            config=self.config,
            rgcn_hidden_feats=hyperparams['rgcn_hidden_feats'],
            ffn_hidden_feats=hyperparams['ffn_hidden_feats'],
            ffn_dropout=hyperparams['ffn_dropout'],
            rgcn_dropout=hyperparams['rgcn_dropout'],
            classification=False,  # Force regression mode
            num_classes=None
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.models.append({
            'model': model,
            'cv': cv_num,
            'fold': fold,
            'hyperparams': hyperparams
        })

    def _postprocess_tg(self, preds: Dict[str, float]) -> Dict[str, float]:
        if not getattr(self.config, 'CALIBRATE_TG', False):
            return preds
        calib_dir = os.path.join(getattr(self.config, 'output_dir', os.path.dirname(self.input_csv)),
                                 getattr(self.config, 'CALIBRATION_OUTDIR', 'calibration/').strip('/'))
        npz = os.path.join(calib_dir, 'tg_quantile_map.npz')
        if os.path.exists(npz) and 'Tg' in preds and preds['Tg'] is not None:
            try:
                z = np.load(npz)
                v = float(preds['Tg'])
                preds['Tg'] = float(np.interp(v, z['pred_q'], z['true_q']))
            except Exception as e:
                self.logger.warning(f'Tg calibration failed at predict time: {e}')
        return preds

    def _load_mu_ffv(self) -> float:
        """Load or compute the training-set mean FFV for margin reporting.
        Falls back to 0.36 if unavailable.
        """
        try:
            # Try to read processed training meta
            meta_path = self.config.get_processed_file_path('meta', 'primary')
            if os.path.exists(meta_path):
                df = pd.read_csv(meta_path)
                if 'group' in df.columns and 'FFV' in df.columns:
                    vals = pd.to_numeric(df.loc[df['group'] == 'training', 'FFV'], errors='coerce').dropna()
                    if len(vals) > 0:
                        mu = float(vals.mean())
                        self.logger.info(f"Loaded training-set mean FFV: {mu:.4f}")
                        return mu
            # Optional: look for simple stats file in output_dir
            stats_path = os.path.join(getattr(self.config, 'output_dir', '.'), 'ffv_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    js = json.load(f)
                    mu = float(js.get('mu_ffv', 0.36))
                    self.logger.info(f"Loaded mu_ffv from stats: {mu:.4f}")
                    return mu
        except Exception as e:
            self.logger.warning(f"mu_ffv unavailable from meta/stats: {e}")
        self.logger.warning("Using default mu_ffv=0.36")
        return 0.36

    def _emit_xai_json(self, comp_id: str, smiles: str, per_target: Dict[str, Dict], summary: Dict[str, Any]):
        """Write a per-molecule XAI JSON and append rollup to _index.jsonl."""
        if not getattr(self.config, 'EMIT_XAI_JSON', True):
            return
        data = {
            'id': comp_id,
            'smiles': smiles,
            # attach representation if available
            'text_repr': self._repr_map.get(comp_id, {}).get('text_repr', smiles),
            'used_bigsmiles': bool(self._repr_map.get(comp_id, {}).get('used_bigsmiles', False)),
            'per_target': per_target,
            'summary': summary
        }
        # Add FFV-focused convenience fields if available
        try:
            ffv = per_target.get('FFV', {})
            if 'prediction' in ffv:
                data['ffv_pred'] = float(ffv['prediction'])
            if 'pi' in ffv and isinstance(ffv['pi'], (list, tuple)):
                # approximate std from PI width if std not provided in summary
                pass
            # Inject std and residual margin if present in summary
            if isinstance(summary.get('ensemble_std'), dict) and summary['ensemble_std'].get('FFV') is not None:
                data['prediction_std'] = float(summary['ensemble_std']['FFV'])
                data['uncertainty_std'] = float(summary['ensemble_std']['FFV'])
            if hasattr(self, 'mu_ffv') and isinstance(self.mu_ffv, (float, int)) and 'ffv_pred' in data:
                data['ffv_margin_from_mean'] = float(data['ffv_pred'] - self.mu_ffv)
            # Tag XAI method
            data['xai_method'] = str(getattr(self.config, 'substructure_type', 'murcko') or 'murcko')
        except Exception:
            pass
        try:
            out_path = os.path.join(self.xai_outdir, f'{comp_id}.json')
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f'Failed to write XAI JSON for {comp_id}: {e}')
        # Append minimal index
        try:
            idx_row = {
                'id': comp_id,
                'FFV_prediction': per_target.get('FFV', {}).get('prediction', None),
                'FFV_agreement@K': per_target.get('FFV', {}).get('agreement', {}).get(f"frag_vs_tok@{getattr(self.config,'XAI_TOPK',10)}", None),
                'uncertainty_flag': summary.get('uncertainty_flag', None)
            }
            with open(self.xai_index_path, 'a') as f:
                f.write(json.dumps(idx_row) + "\n")
        except Exception as e:
            self.logger.warning(f'Failed to write XAI index for {comp_id}: {e}')
    


    def save_checkpoint(self, state: Dict, checkpoint_path: str = None):
        """
        Save checkpoint for recovery.
        
        Args:
            state (Dict): State to save in the checkpoint
            checkpoint_path (str, optional): Custom path to save checkpoint. 
                                            Defaults to self.checkpoint_file.
        """
        try:
            # Use provided path or default to the standard location
            save_path = checkpoint_path if checkpoint_path else self.checkpoint_file
            
            # Add metadata to state
            state['metadata'] = {
                'input_csv': self.input_csv,
                'timestamp': datetime.now().isoformat(),
                'total_molecules': state.get('total_molecules', 0),
                'start_idx': state.get('start_idx', 0),
                'end_idx': state.get('end_idx', None)
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            self.logger.info(f"Saved checkpoint to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
    


    def load_checkpoint(self, checkpoint_path: str = None) -> Dict:
        """
        Load checkpoint if exists and validate it matches current input file.
        
        Args:
            checkpoint_path (str, optional): Custom path to load checkpoint from.
                                        Defaults to self.checkpoint_file.
                                        
        Returns:
            Dict: Checkpoint data or empty template if no valid checkpoint
        """
        # Use provided path or default
        load_path = checkpoint_path if checkpoint_path else self.checkpoint_file
        
        if os.path.exists(load_path):
            try:
                with open(load_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                # Validate checkpoint matches current run
                metadata = checkpoint.get('metadata', {})
                if metadata.get('input_csv') != self.input_csv:
                    self.logger.warning(f"Checkpoint at {load_path} is from a different input file. Starting fresh.")
                    return {'current_idx': 0, 'results': []}
                
                # Log which specific checkpoint was loaded
                self.logger.info(f"Loaded checkpoint from {load_path}")
                
                # Return the checkpoint data
                return checkpoint
            except Exception as e:
                self.logger.error(f"Error loading checkpoint from {load_path}: {str(e)}")
                return {'current_idx': 0, 'results': []}
        else:
            # Log only if it would have been expected to exist
            if checkpoint_path:
                self.logger.info(f"No checkpoint found at {load_path}")
            return {'current_idx': 0, 'results': []}
        


    def _predict_single_model(self, model_info: Dict, graph: Data) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
        """Make regression prediction for a single model returning all properties."""
        try:
            model = model_info['model']
            with torch.inference_mode():
                with GPUMemoryGuard():
                    # Create a new graph object with tensors on the correct device
                    graph_device = next(model.parameters()).device
                    graph_copy = Data(
                        x=graph.x.to(graph_device),
                        edge_index=graph.edge_index.to(graph_device),
                        edge_attr=graph.edge_attr.to(graph_device) if hasattr(graph, 'edge_attr') else None,
                        edge_type=graph.edge_type.to(graph_device) if hasattr(graph, 'edge_type') else None,
                        batch=None  # Will be set by Batch.from_data_list
                    )
                    
                    if hasattr(graph, 'smask'):
                        graph_copy.smask = graph.smask.to(graph_device)
                    
                    # Create batch with tensor on the correct device
                    model_batch = Batch.from_data_list([graph_copy])
                    
                    # Ensure model is in eval mode
                    model.eval()
                    
                    # Make prediction - this should return a dict for regression
                    pred_dict, atom_weights = model(model_batch)
                
                # Handle the prediction output
                if isinstance(pred_dict, dict):
                    # Multi-property regression - extract all properties
                    preds = {}
                    for prop in PROPERTY_NAMES:
                        if prop in pred_dict:
                            tensor_val = pred_dict[prop]
                            if hasattr(tensor_val, 'item'):
                                preds[prop] = float(tensor_val.item())
                            else:
                                preds[prop] = float(tensor_val)
                        else:
                            self.logger.warning(f"Property {prop} not found in model output")
                            preds[prop] = 0.0
                else:
                    # Single output case - shouldn't happen but handle it
                    self.logger.warning("Model returned single output instead of property dictionary")
                    if hasattr(pred_dict, 'item'):
                        val = float(pred_dict.item())
                    else:
                        val = float(pred_dict)
                    # Distribute to first property or create default
                    preds = {PROPERTY_NAMES[0]: val}
                    for prop in PROPERTY_NAMES[1:]:
                        preds[prop] = 0.0

                # Move atom weights to CPU if they exist
                atom_weights_np = atom_weights.cpu().numpy() if atom_weights is not None else None
                
                return preds, atom_weights_np
                    
        except Exception as e:
            self.logger.error(f"Error in model prediction: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, None

    def _predict_single_model_regression(self, model_info: Dict, smiles: str) -> Optional[Dict[str, float]]:
        """Get regression predictions for all properties for one model."""
        try:
            graph = self._get_graph(smiles, mask=[])
            if graph is None:
                return None
            model = model_info["model"]
            model.eval()
            with torch.inference_mode():
                with GPUMemoryGuard():
                    batch = Batch.from_data_list([graph]).to(self.device)
                    preds, _ = model(batch)
            result = {p: float(preds[p].cpu().item()) for p in PROPERTY_NAMES if p in preds}
            # Explicitly clear references
            try:
                batch = None
                graph = None
            except Exception:
                pass
            return result
        except Exception as e:
            self.logger.error(f"Error in regression prediction: {e}")
            return None

    def _predict_single_model_regression_masked(self, model_info: Dict, smiles: str, atoms: List[int]) -> Optional[Dict[str, float]]:
        """Get regression prediction for a masked substructure."""
        try:
            graph = self._get_graph(smiles, mask=atoms)
            if graph is None:
                return None
            model = model_info["model"]
            model.eval()
            with torch.inference_mode():
                with GPUMemoryGuard():
                    batch = Batch.from_data_list([graph]).to(self.device)
                    preds, _ = model(batch)
            result = {p: float(preds[p].cpu().item()) for p in PROPERTY_NAMES if p in preds}
            try:
                batch = None
                graph = None
            except Exception:
                pass
            return result
        except Exception as e:
            self.logger.error(f"Error in masked regression prediction: {e}")
            return None

    def determine_confidence_thresholds(self, property_stds_dataset: List[Dict[str, float]]):
        """Determine confidence thresholds dynamically from dataset analysis."""
        thresholds = {}
        for prop in PROPERTY_NAMES:
            prop_stds = [d.get(f"{prop}_std") for d in property_stds_dataset if d.get(f"{prop}_std") is not None]
            if prop_stds:
                threshold = np.percentile(prop_stds, 75)
                self.logger.info(f"{prop} confidence threshold (75th percentile): {threshold:.4f}")
                thresholds[prop] = threshold
            else:
                fallback = {"Tg": 10.0, "FFV": 0.009, "Tc": 0.008, "Density": 0.017, "Rg": 0.4}
                thresholds[prop] = fallback[prop]
                self.logger.warning(f"Using fallback threshold for {prop}: {thresholds[prop]}")
        return thresholds

    def analyze_dataset_confidence(self, input_csv: str, sample_size: int = 1000):
        """Analyze dataset subset to compute confidence thresholds."""
        df = pd.read_csv(input_csv) if input_csv.endswith(".csv") else pd.read_parquet(input_csv)
        sample_df = df.sample(n=min(sample_size, len(df)))
        sample_stds = []
        for _, row in sample_df.iterrows():
            smiles = row["SMILES"]
            property_preds = {p: [] for p in PROPERTY_NAMES}
            for model_info in self.models:
                preds = self._predict_single_model_regression(model_info, smiles)
                if preds:
                    for prop, val in preds.items():
                        property_preds[prop].append(val)

            mol_stds = {}
            for prop, vals in property_preds.items():
                if len(vals) > 1:
                    mol_stds[f"{prop}_std"] = np.std(vals)
                    mol_stds[f"{prop}_mean"] = np.mean(vals)

            if mol_stds:
                sample_stds.append(mol_stds)
        return self.determine_confidence_thresholds(sample_stds)


    def predict_molecule_with_confidence(self, smiles: str, confidence_thresholds: Optional[Dict[str, float]] = None) -> Dict:
        """Predict molecule properties and perform confidence-based XAI."""
        all_predictions = {p: [] for p in PROPERTY_NAMES}
        for model_info in self.models:
            preds = self._predict_single_model_regression(model_info, smiles)
            if preds:
                for prop, val in preds.items():
                    all_predictions[prop].append(val)

        result = {"SMILES": smiles}
        high_conf_props = []
        
        # Calculate predictions and confidence for each property
        for prop in PROPERTY_NAMES:
            vals = all_predictions[prop]
            if vals:
                mean_pred = np.mean(vals)
                std_pred = np.std(vals)
                result[f"pred_{prop}"] = float(mean_pred)
                result[f"{prop}_std"] = float(std_pred)
                
                if confidence_thresholds and prop in confidence_thresholds:
                    thr = confidence_thresholds[prop]
                    confident = std_pred <= thr
                    result[f"{prop}_confidence"] = "high" if confident else "low"
                    if confident:
                        high_conf_props.append(prop)
                else:
                    result[f"{prop}_confidence"] = "unknown"

        result["strategic_wmae_estimate"] = calculate_strategic_wmae(result)
        # Add FFV-centric convenience fields
        try:
            ffv = result.get('pred_FFV')
            if ffv is not None:
                if getattr(self.config, 'CALIBRATE_FFV', False) and self.ffv_calibrator is not None:
                    try:
                        result['pred_FFV_calibrated'] = float(self.ffv_calibrator.predict([float(ffv)])[0])
                    except Exception:
                        pass
                result['ffv_pred'] = float(result.get('pred_FFV_calibrated', ffv))
                result['prediction_std'] = float(result.get('FFV_std')) if result.get('FFV_std') is not None else None
                if hasattr(self, 'mu_ffv'):
                    result['ffv_margin_from_mean'] = float(result['ffv_pred'] - self.mu_ffv)
        except Exception:
            pass
        result['n_models'] = len([m for m in self.models if any(all_predictions[p] for p in PROPERTY_NAMES)])

        # XAI analysis based on confidence and substructure type
        sub_type = getattr(self.config, "substructure_type", None)

        # Check if we should compute XAI for all molecules (bypass confidence gating)
        compute_for_all = getattr(self.config, 'COMPUTE_XAI_FOR_ALL', False)

        # Determine which properties to analyze
        if compute_for_all:
            # Compute XAI for primary target regardless of confidence
            props_to_analyze = [getattr(self.config, 'PRIMARY_TARGET', 'FFV')]
        else:
            # Use high confidence properties only
            props_to_analyze = high_conf_props

        if sub_type and props_to_analyze:
            try:
                # Perform XAI analysis
                for prop in props_to_analyze:
                    attrs = self.analyze_property_attributions(smiles, prop)
                    for i, (attr, smi) in enumerate(attrs.get("scaffolds", [])):
                        result[f"{sub_type}_substructure_{i}_{prop}_attribution"] = attr
                        result[f"{sub_type}_substructure_{i}_smiles"] = smi
                result["xai_status"] = "Completed"
            except Exception as e:
                self.logger.error(f"Error in XAI analysis: {e}")
                result["xai_status"] = f"Error - {e}"
        else:
            if not sub_type:
                result["xai_status"] = "Skipped - No substructure type"
            else:
                result["xai_status"] = "Skipped - Low model confidence"

        return result

    def analyze_property_attributions(self, smiles: str, target_property: str) -> Dict[str, Any]:
        """Calculate attributions for a specific property."""
        substructures = self.get_substructures(smiles, self.config.substructure_type)
        if not substructures:
            return {"scaffolds": [], "substituents": []}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"scaffolds": [], "substituents": []}

        scaffold_attrs = []
        scaffold_smiles = []
        for idx, atoms in substructures.items():
            try:
                scf_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), kekuleSmiles=True)
                if ":" in scf_smi or "*" in scf_smi:
                    continue
                diffs = []
                for model_info in self.models:
                    orig = self._predict_single_model_regression(model_info, smiles)
                    masked = self._predict_single_model_regression_masked(model_info, smiles, atoms)
                    if orig and masked and target_property in orig and target_property in masked:
                        diffs.append(orig[target_property] - masked[target_property])
                if diffs:
                    scaffold_attrs.append(float(np.mean(diffs)))
                    scaffold_smiles.append(scf_smi)
            except Exception as e:
                self.logger.debug(f"Error processing substructure {idx}: {e}")
                continue

        if scaffold_attrs:
            norm = scaffold_attrs
            return {"scaffolds": list(zip(norm, scaffold_smiles))}
        return {"scaffolds": [], "substituents": []}
    







    def predict_molecule(self, smiles: str) -> Dict:
        """
        Predict properties for a single molecule using ensemble averaging,
        and include scaffold and substituent attributions in a flat structure.
        """
        try:
            # Construct base graph
            graph = self._get_graph(smiles, mask=[])
            if graph is None:
                return {'error': 'Failed to construct graph', 'SMILES': smiles}

            # Ensemble predictions - collect all property predictions
            all_predictions = {prop: [] for prop in PROPERTY_NAMES}
            atom_weights_list = []
            
            for model_info in self.models:
                pred_dict, atom_w = self._predict_single_model(model_info, graph)
                if pred_dict is not None:
                    # Add predictions for each property
                    for prop in PROPERTY_NAMES:
                        if prop in pred_dict:
                            all_predictions[prop].append(pred_dict[prop])
                    
                    if atom_w is not None:
                        atom_weights_list.append(atom_w)

            # Check if we got any predictions
            if not any(all_predictions.values()):
                return {'error': 'No valid predictions', 'SMILES': smiles}

            # Calculate ensemble statistics for each property
            result = {'SMILES': smiles}
            valid_models = 0
            
            for prop in PROPERTY_NAMES:
                vals = all_predictions[prop]
                if vals:
                    result[f'pred_{prop}'] = float(np.mean(vals))
                    result[f'{prop}_std'] = float(np.std(vals))
                    valid_models = max(valid_models, len(vals))

            # Optional Tg calibration (post-process)
            calced = {p: result.get(f'pred_{p}') for p in PROPERTY_NAMES}
            calced = self._postprocess_tg(calced)
            for p, v in calced.items():
                if v is not None:
                    result[f'pred_{p}'] = v
            
            result['n_models'] = valid_models

            # Include atom-level weights if available
            if atom_weights_list:
                result['atom_weights'] = np.mean(atom_weights_list, axis=0).tolist()

            # Add FFV-centric convenience fields
            try:
                ffv = result.get('pred_FFV')
                if ffv is not None:
                    if getattr(self.config, 'CALIBRATE_FFV', False) and self.ffv_calibrator is not None:
                        try:
                            result['pred_FFV_calibrated'] = float(self.ffv_calibrator.predict([float(ffv)])[0])
                        except Exception:
                            pass
                    result['ffv_pred'] = float(result.get('pred_FFV_calibrated', ffv))
                    result['prediction_std'] = float(result.get('FFV_std')) if result.get('FFV_std') is not None else None
                    if hasattr(self, 'mu_ffv'):
                        result['ffv_margin_from_mean'] = float(result['ffv_pred'] - self.mu_ffv)
            except Exception:
                pass

            # Substructure analysis
            subtype = getattr(self.config, 'substructure_type', None)
            if subtype:
                try:
                    t = subtype
                    sc_attrs, sc_smis, sc_subs = self.analyze_substructures(smiles, subtype)
                    
                    # Add scaffold information
                    for i, (attr, smi) in enumerate(zip(sc_attrs, sc_smis)):
                        result[f'{t}_substructure_{i}_attribution'] = attr
                        result[f'{t}_substructure_{i}_smiles'] = smi
                        
                        # Flatten the substituents data
                        if i < len(sc_subs):
                            for j, sub in enumerate(sc_subs[i]):
                                result[f'{t}_substituent_{i}_{j}_smiles'] = sub.get('smiles', '')
                                result[f'{t}_substituent_{i}_{j}_context'] = sub.get('context', '')
                                result[f'{t}_substituent_{i}_{j}_attribution'] = sub.get('attribution', 0.0)
                except Exception as e:
                    self.logger.error(f"Error in substructure analysis: {str(e)}")
                    result['substructure_error'] = str(e)

            return result

        except Exception as e:
            self.logger.error(f"Error predicting molecule {smiles}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e), 'SMILES': smiles}






    def analyze_molecule_scaffolds(self, smiles: str) -> Dict:
        """Analyze molecule's scaffold and substituents"""
        try:
            # Get Murcko scaffolds
            substructures = self.get_substructures(smiles, 'murcko')
            
            if not substructures:
                return {"error": "No scaffolds found", "SMILES": smiles}
            
            # Create a formatted result with clear hierarchy
            result = {
                "SMILES": smiles,
                "num_scaffolds": len(substructures),
                "num_substituents": 0  # Will be calculated below
            }
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES", "SMILES": smiles}
            
            # Add scaffolds
            for i, (idx, atoms) in enumerate(substructures.items()):
                try:
                    scaffold_smiles = Chem.MolFragmentToSmiles(
                        mol, atomsToUse=list(atoms), kekuleSmiles=True
                    )
                    result[f"scaffold_{i+1}_smiles"] = scaffold_smiles
                except Exception as e:
                    self.logger.debug(f"Error getting scaffold SMILES: {e}")
                    result[f"scaffold_{i+1}_smiles"] = "ERROR"
            
            # Add basic substituent analysis
            all_scaffold_atoms = set()
            for atoms in substructures.values():
                all_scaffold_atoms.update(atoms)
            
            # Find substituents (atoms not in any scaffold)
            all_atoms = set(range(mol.GetNumAtoms()))
            substituent_atoms = all_atoms - all_scaffold_atoms
            
            if substituent_atoms:
                # Group substituent atoms into connected components
                substituents = []
                visited = set()
                
                for atom_idx in substituent_atoms:
                    if atom_idx in visited:
                        continue
                        
                    # BFS to find connected substituent
                    component = set()
                    queue = [atom_idx]
                    
                    while queue:
                        current = queue.pop(0)
                        if current in visited:
                            continue
                        visited.add(current)
                        component.add(current)
                        
                        # Add connected neighbors that are also substituents
                        atom = mol.GetAtomWithIdx(current)
                        for neighbor in atom.GetNeighbors():
                            neighbor_idx = neighbor.GetIdx()
                            if neighbor_idx in substituent_atoms and neighbor_idx not in visited:
                                queue.append(neighbor_idx)
                    
                    if component:
                        substituents.append(list(component))
                
                # Add substituent information
                result["num_substituents"] = len(substituents)
                for i, sub_atoms in enumerate(substituents):
                    try:
                        sub_smiles = Chem.MolFragmentToSmiles(
                            mol, atomsToUse=sub_atoms, kekuleSmiles=True
                        )
                        result[f"substituent_{i+1}_smiles"] = sub_smiles
                        result[f"substituent_{i+1}_attachment"] = "unknown"
                        result[f"substituent_{i+1}_r_group"] = f"R{i+1}"
                    except Exception as e:
                        self.logger.debug(f"Error getting substituent SMILES: {e}")
                        result[f"substituent_{i+1}_smiles"] = "ERROR"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing scaffolds for {smiles}: {str(e)}")
            return {"error": str(e), "SMILES": smiles}









    def get_substructures(self, smiles: str, substructure_type: str) -> dict:
        """
        Get substructures based on type. For 'murcko', use a fast polymer-aware mask builder.
        """
        try:
            # Fast polymer-aware Murcko path (replaces 40-combo enumeration)
            if substructure_type == 'murcko':
                mol = self._get_mol(smiles)
                if mol is None:
                    return {}
                candidates = []
                for atoms in self.build_polymer_masks(mol):
                    if atoms:
                        candidates.append({'mask_id': 'POLY', 'atoms': list(map(int, atoms)), 'mask_family': 'fragment'})
                # finalize using existing disjoint selector
                topk = int(getattr(self.config, 'XAI_TOP_SCAFFOLDS', getattr(self.config, 'XAI_TOPK', 10)))
                selected = self._select_non_overlapping_masks(candidates, topk)
                return {i: m['atoms'] for i, m in enumerate(selected)}
            if substructure_type == 'murcko':
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return {}

                # build adjacency list for BFS
                graph = {
                    atom.GetIdx(): [nbr.GetIdx() for nbr in atom.GetNeighbors()]
                    for atom in mol.GetAtoms()
                }

                # get all rings as sets of atom‐indices
                ring_sets = [set(r) for r in mol.GetRingInfo().AtomRings()]
                if not ring_sets:
                    return {}

                def shortest_path(src: set, dst: set):
                    visited = set(src)
                    queue = deque([[i] for i in src])
                    while queue:
                        path = queue.popleft()
                        last = path[-1]
                        if last in dst:
                            return path
                        for nb in graph[last]:
                            if nb not in visited:
                                visited.add(nb)
                                queue.append(path + [nb])
                    return []

                # enumerate all combinations of rings 1..N
                all_scaffolds = []
                N = len(ring_sets)
                for k in range(1, N + 1):
                    for combo in itertools.combinations(range(N), k):
                        atoms = set().union(*(ring_sets[i] for i in combo))
                        # if more than one ring, connect them
                        if len(combo) > 1:
                            base = ring_sets[combo[0]]
                            for idx in combo[1:]:
                                atoms |= set(shortest_path(base, ring_sets[idx]))
                        all_scaffolds.append(atoms)

                # dedupe
                unique = []
                for s in all_scaffolds:
                    if not any(s == u for u in unique):
                        unique.append(s)

                # build indexed dict
                scaffolds = {i: sorted(s) for i, s in enumerate(unique)}

                # now **limit** to top 40 by size
                MAX_SCAFFOLDS = 40
                if len(scaffolds) > MAX_SCAFFOLDS:
                    # sort indices by scaffold‐size descending
                    sorted_idxs = sorted(
                        scaffolds.keys(),
                        key=lambda i: len(scaffolds[i]),
                        reverse=True
                    )[:MAX_SCAFFOLDS]
                    scaffolds = {i: scaffolds[i] for i in sorted_idxs}

                return scaffolds

            elif substructure_type == 'brics':
                # … your existing BRICS code …
                info = return_brics_leaf_structure(smiles)
                return info.get('substructure', {})

            elif substructure_type == 'fg':
                # … your existing FG code …
                if not hasattr(self.config, 'fg_with_ca_list'):
                    self._initialize_fg_patterns()
                fg_hits, _ = return_fg_hit_atom(
                    smiles,
                    self.config.fg_name_list,
                    self.config.fg_with_ca_list,
                    self.config.fg_without_ca_list
                )
                return {i: hits[0] for i, hits in enumerate(fg_hits) if hits}

            else:
                self.logger.warning(
                    f"No substructures for type '{substructure_type}' in {smiles}"
                )
                return {}

        except Exception as e:
            self.logger.error(f"Error extracting [{substructure_type}] for {smiles}: {e}")
            return {}







    def _extract_substituents_with_context(self, smiles: str, core_atoms: List[int]) -> List[Dict[str,Any]]:
        #print(f"DEBUG: _extract_substituents_with_context called with {len(core_atoms)} core atoms")
        """
        Identify true medicinal chemistry substituents - terminal groups attached to a core scaffold.
        
        Args:
            smiles: Molecule SMILES string
            core_atoms: List of atom indices that belong to the core scaffold
            
        Returns:
            List of dictionaries containing substituent information
        """
        mol = self._get_mol(smiles)
        if mol is None:
            return []

        core_set = set(core_atoms)
        all_idx = set(range(mol.GetNumAtoms()))
        non_core_idx = all_idx - core_set

        # Build atom connectivity map
        nbrs = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()]
                for a in mol.GetAtoms()}
        
        # Find attachment points: core atoms connected to non-core atoms
        attachment_points = {}
        for core_atom in core_set:
            for nbr in nbrs[core_atom]:
                if nbr in non_core_idx:
                    if nbr not in attachment_points:
                        attachment_points[nbr] = core_atom

        # Collect substituent fragments
        visited = set()
        true_substituents = []
        
        for start_atom in attachment_points.keys():
            if start_atom in visited:
                continue
                
            # Do a BFS to find the connected substituent fragment
            fragment = set()
            queue = deque([start_atom])
            
            while queue:
                atom_idx = queue.popleft()
                fragment.add(atom_idx)
                visited.add(atom_idx)
                
                for neighbor in nbrs[atom_idx]:
                    if neighbor in non_core_idx and neighbor not in fragment and neighbor not in visited:
                        queue.append(neighbor)
            
            # Get the attachment point
            attachment_atom = attachment_points[start_atom]
            
            # Determine if the attachment point is aromatic
            is_aromatic = mol.GetAtomWithIdx(attachment_atom).GetIsAromatic()
            context = 'aromatic' if is_aromatic else 'aliphatic'
            
            # Get SMILES for this fragment
            try:
                frag_smiles = Chem.MolFragmentToSmiles(
                    mol, atomsToUse=list(fragment), kekuleSmiles=True
                )
                
                # Filter out suspicious fragments that are likely parts of rings
                suspicious_fragments = ['CCC', 'CC', 'C', 'c']
                if frag_smiles in suspicious_fragments:
                    # Extra check: if this fragment is in a ring, skip it
                    in_ring = any(mol.GetAtomWithIdx(idx).IsInRing() for idx in fragment)
                    if in_ring:
                        continue
                    
                true_substituents.append({
                    'fragment_atoms': list(fragment),
                    'smiles': frag_smiles,
                    'attachment_atom': attachment_atom,
                    'context': context
                })
            except Exception as e:
                self.logger.warning(f"Failed to process substituent: {e}")
                
        return true_substituents


    





    def attribute_scaffolds(self, smiles: str, substructure_type: str) -> Tuple[List[float], List[str], List[List[Dict[str,Any]]]]:
        """
        Analyze substructures with adaptive scaling for attributions.
        
        Args:
            smiles: Molecule SMILES string
            substructure_type: Type of substructure to analyze ('murcko', 'brics', 'fg')
        
        Returns:
            Tuple[List[float], List[str], List[List[Dict[str,Any]]]]: 
                - Per-scaffold attribution scores
                - Scaffold SMILES
                - List of substituents (with smiles, context, attribution)
        """
        import math
        
        subs_dict = self.get_substructures(smiles, substructure_type)
        if not subs_dict:
            return [], [], []

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [], [], []

        # 1) Collect all logit differences for all substructures (scaffolds)
        scaffold_diffs = []
        scaffold_atoms = []
        scaffold_smis = []
        
        for idx, atoms in subs_dict.items():
            atoms = list(map(int, atoms))
            
            # Get scaffold SMILES (skip weird ones)
            try:
                scf_smi = Chem.MolFragmentToSmiles(
                    mol, atomsToUse=atoms, kekuleSmiles=True
                )
                if ':' in scf_smi or '*' in scf_smi:
                    continue
            except:
                continue
            
            # Collect logit differences from all models for this scaffold
            model_diffs = []
            for minfo in self.models:
                _, diff = self.calculate_attribution(
                    minfo['model'], smiles, atoms, self.device
                )
                if diff is not None:
                    model_diffs.append(diff)

            # Validation check for first scaffold only
            if idx == 0 and model_diffs:
                try:
                    # Manual validation by recalculating the first model's prediction
                    first_model = self.models[0]['model']
                    original_graph = self._get_graph(smiles, mask=[])
                    masked_graph = self._get_graph(smiles, mask=atoms)
                    
                    if original_graph is not None and masked_graph is not None:
                        with torch.no_grad():
                            original_batch = Batch.from_data_list([original_graph]).to(self.device)
                            masked_batch = Batch.from_data_list([masked_graph]).to(self.device)
                            
                            original_pred, _ = first_model(original_batch)
                            masked_pred, _ = first_model(masked_batch)
                            
                            manual_diff = original_pred.item() - masked_pred.item()
                            self.logger.debug(
                                f"First scaffold manual check: {manual_diff:.6f}, pipeline {model_diffs[0]:.6f}"
                            )
                except Exception as e:
                    self.logger.debug(f"Manual check failed: {e}")
            
            if not model_diffs:
                continue
                
            mean_diff = float(np.mean(model_diffs))
            scaffold_diffs.append(mean_diff)
            scaffold_atoms.append(atoms)
            scaffold_smis.append(scf_smi)
        
        if not scaffold_diffs:
            return [], [], []
        
        # 2) Gather substituent differences 
        substituent_raw = []

        for atoms in scaffold_atoms:
            raw_subs = self._extract_substituents_with_context(smiles, atoms)
            sub_entries = []

            for sub in raw_subs:
                sub_diffs = []
                for minfo in self.models:
                    _, sub_diff = self.calculate_attribution(
                        minfo['model'], smiles, sub['fragment_atoms'], self.device
                    )
                    if sub_diff is not None:
                        sub_diffs.append(sub_diff)

                if sub_diffs:
                    mean_sub_diff = float(np.mean(sub_diffs))
                    sub_entries.append({
                        'smiles': sub['smiles'],
                        'context': sub['context'],
                        'diff': mean_sub_diff
                    })

            substituent_raw.append(sub_entries)

        # Log substituent collection
        self.logger.debug(
            f"Scaffolds collected: {len(scaffold_diffs)}, "
            f"substituents: {sum(len(e) for e in substituent_raw)}"
        )

        # Collect all differences for scaling
        all_diffs = list(scaffold_diffs) + [sub['diff'] for entries in substituent_raw for sub in entries]

        if not all_diffs:
            return [], [], []

        # Aggregate attributions at atom level and recompute fragment scores
        frag_atoms_all = scaffold_atoms + [sub['fragment_atoms'] for entries in substituent_raw for sub in entries]
        frag_diffs_all = scaffold_diffs + [sub['diff'] for entries in substituent_raw for sub in entries]
        _, corrected_all = atom_disjoint_attribution(mol, frag_atoms_all, frag_diffs_all)
        
        # Update with corrected values
        scaffold_diffs = corrected_all[:len(scaffold_diffs)]
        corr_idx = len(scaffold_diffs)
        for entries in substituent_raw:
            for sub in entries:
                sub['diff'] = corrected_all[corr_idx]
                corr_idx += 1

        # Update all_diffs with corrected values
        all_diffs = list(scaffold_diffs) + [sub['diff'] for entries in substituent_raw for sub in entries]

        self.logger.debug(
            f"Raw differences before scaling: min={min(all_diffs):.4f}, max={max(all_diffs):.4f}"
        )

        # no need for scaling in regression ulike binary classification
        scaled_all = all_diffs
        scaffold_attrs = scaled_all[:len(scaffold_diffs)]
        start_idx = len(scaffold_diffs)

        # Build final substituent list with attributions
        all_subs = []
        for entries in substituent_raw:
            subs_with_attr = []
            for sub in entries:
                sub_smiles = sub['smiles']
                if ':' in sub_smiles:
                    sub_smiles = sub_smiles.replace(":", "-")
                    sub['smiles'] = sub_smiles

                subs_with_attr.append({
                    'smiles': sub['smiles'],
                    'context': sub['context'],
                    'attribution': scaled_all[start_idx]
                })
                start_idx += 1

            all_subs.append(subs_with_attr)

        # Debug logging
        raw_sum = sum(scaffold_diffs) + sum(v['diff'] for entries in substituent_raw for v in entries)
        scaled_sum = sum(scaffold_attrs) + sum(v['attribution'] for lst in all_subs for v in lst)
        if scaffold_smis:
            self.logger.debug(
                f"First scaffold {scaffold_smis[0]} raw={scaffold_diffs[0]:.4f}, scaled={scaffold_attrs[0]:.4f}"
            )
        self.logger.debug(
            f"Scale factor raw_max={max(abs(d) for d in all_diffs):.4f} -> scaled range=({min(scaled_all):.4f},{max(scaled_all):.4f})"
        )
        self.logger.debug(f"Conservation check raw_sum={raw_sum:.4f} scaled_sum={scaled_sum:.4f}")

        return scaffold_attrs, scaffold_smis, all_subs








    def analyze_substructures(
        self, 
        smiles: str, 
        substructure_type: str
    ) -> Tuple[List[float], List[str], List[List[Dict[str,Any]]]]:
        """
        Analyze substructures for attribution with improved batching and caching.
        
        Returns:
            Tuple of (attributions, SMILES, substituents)
        """
        # Use caching to avoid redundant calculations
        if not hasattr(self, '_analysis_cache'):
            self._analysis_cache = {}
            
        cache_key = f"{smiles}_{substructure_type}" 
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        try:
            # Get substructures with optimized extraction
            subs_dict = self.get_substructures(smiles, substructure_type)
            if not subs_dict:
                return [], [], []

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [], [], []

            # Extract substructure atoms and compute SMILES once
            substructure_atoms = []
            scaffold_smis = []
            
            for idx, atoms in subs_dict.items():
                try:
                    # Ensure atoms is a list of integers
                    atoms = list(map(int, atoms)) if not all(isinstance(a, int) for a in atoms) else atoms
                    
                    scf_smi = Chem.MolFragmentToSmiles(
                        mol, atomsToUse=atoms, kekuleSmiles=True
                    )
                    # Skip invalid SMILES (keep wildcards for polymers)
                    if ':' in scf_smi:
                        continue
                    # Note: Keeping '*' wildcards for polymer connection points
                    substructure_atoms.append(atoms)
                    scaffold_smis.append(scf_smi)
                except Exception as e:
                    self.logger.debug(f"Error getting SMILES for substructure: {str(e)}")
                    continue
            
            if not substructure_atoms:
                return [], [], []
            
            # Calculate attributions for all substructures
            all_model_diffs = []
            
            for model_info in self.models:
                # Calculate all attributions in one batch per model
                orig_pred, attributions = self.calculate_attribution(
                    model_info['model'],
                    smiles,
                    substructure_atoms,
                    self.device
                )

                if attributions:
                    all_model_diffs.append(attributions)
            
            # Average attribution differences across models for PRIMARY_TARGET only
            scaffold_diffs = []
            target_prop = getattr(self.config, 'PRIMARY_TARGET', 'FFV')
            for atoms in substructure_atoms:
                atom_tuple = tuple(atoms)
                
                # For multi-property regression, we need to handle dict attributions
                diffs = []
                for model_diffs in all_model_diffs:
                    if atom_tuple in model_diffs:
                        diff_val = model_diffs[atom_tuple]
                        if isinstance(diff_val, dict):
                            # Use only target property
                            if target_prop in diff_val:
                                diffs.append(float(diff_val.get(target_prop, 0.0)))
                        elif diff_val is not None:
                            diffs.append(diff_val)
                
                # Calculate average difference
                if diffs:
                    avg_diff = float(np.mean(diffs))
                    scaffold_diffs.append(avg_diff)
                else:
                    scaffold_diffs.append(0.0)
            
            if not scaffold_diffs:
                return [], [], []
            
            # Collect substituents for each scaffold (limit per scaffold), then batch-attribute per model
            substituent_entries: List[List[Dict[str, Any]]] = []
            all_sub_atoms: List[Tuple[int, ...]] = []
            subs_meta: Dict[Tuple[int, ...], Tuple[str, str]] = {}
            per_scaffold_sub_keys: List[List[Tuple[int, ...]]] = []
            max_subs = int(getattr(self.config, 'XAI_TOP_SUBS_PER_SCAFFOLD', 3))

            for atoms in substructure_atoms:
                raw_subs = self._extract_substituents_with_context(smiles, atoms)
                # pick up to max_subs largest by fragment size
                raw_subs = sorted(raw_subs, key=lambda s: -len(s.get('fragment_atoms', [])))[:max_subs]
                sub_keys: List[Tuple[int, ...]] = []
                for sub in raw_subs:
                    key = tuple(sorted(map(int, sub.get('fragment_atoms', []))))
                    if not key:
                        continue
                    sub_keys.append(key)
                    if key not in subs_meta:
                        subs_meta[key] = (sub.get('smiles', ''), sub.get('context', ''))
                        all_sub_atoms.append(key)
                per_scaffold_sub_keys.append(sub_keys)

            # Batch attribution per model for all unique substituents
            model_sub_diffs: List[Dict[Tuple[int, ...], float]] = []
            if all_sub_atoms:
                atom_lists = [list(k) for k in all_sub_atoms]
                for minfo in self.models:
                    _, batch_diffs = self.calculate_attribution(
                        minfo['model'], smiles, atom_lists, self.device
                    )
                    # normalize into dict of target diffs
                    diffs_map: Dict[Tuple[int, ...], float] = {}
                    if batch_diffs:
                        for k in all_sub_atoms:
                            d = batch_diffs.get(tuple(k))
                            if isinstance(d, dict):
                                diffs_map[tuple(k)] = float(d.get(target_prop, 0.0))
                            elif d is not None:
                                diffs_map[tuple(k)] = float(d)
                    model_sub_diffs.append(diffs_map)

            # Aggregate across models and rebuild per-scaffold entries
            for sub_keys in per_scaffold_sub_keys:
                sub_entries: List[Dict[str, Any]] = []
                for key in sub_keys:
                    vals = [m.get(key) for m in model_sub_diffs if key in m]
                    if vals:
                        mean_sub = float(np.mean(vals))
                        smi, ctx = subs_meta.get(key, ('', ''))
                        sub_entries.append({'smiles': smi, 'context': ctx, 'diff': mean_sub, 'fragment_atoms': list(key)})
                substituent_entries.append(sub_entries)

            # Apply atom-disjoint aggregation and normalization
            all_diffs = list(scaffold_diffs) + [sub['diff'] for entries in substituent_entries for sub in entries]

            if not all_diffs:
                return [], [], []

            # Aggregate attributions at atom level and recompute fragment scores
            frag_atoms_all = substructure_atoms + [sub['fragment_atoms'] for entries in substituent_entries for sub in entries]
            frag_diffs_all = scaffold_diffs + [sub['diff'] for entries in substituent_entries for sub in entries]
            _, corrected_all = atom_disjoint_attribution(mol, frag_atoms_all, frag_diffs_all)
            
            # Update with corrected values
            scaffold_diffs = corrected_all[:len(scaffold_diffs)]
            corr_idx = len(scaffold_diffs)
            
            for entries in substituent_entries:
                for sub in entries:
                    sub['diff'] = corrected_all[corr_idx]
                    corr_idx += 1

            # Apply normalization (identity scaling, keep sign and magnitude)
            all_diffs = list(scaffold_diffs) + [sub['diff'] for entries in substituent_entries for sub in entries]
            scaled_all = all_diffs
            scaffold_attrs = scaled_all[:len(scaffold_diffs)]
            start_idx = len(scaffold_diffs)

            # Build final substituent results
            all_subs = []
            for entries in substituent_entries:
                subs_with_attr = []
                for sub in entries:
                    sub_smiles = sub['smiles']
                    if ':' in sub_smiles:
                        sub_smiles = sub_smiles.replace(":", "-")

                    subs_with_attr.append({
                        'smiles': sub_smiles,
                        'context': sub['context'],
                        'attribution': scaled_all[start_idx]
                    })
                    start_idx += 1

                all_subs.append(subs_with_attr)
            
            # Cache results
            result = (scaffold_attrs, scaffold_smis, all_subs)
            self._analysis_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in substructure analysis for {smiles}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return [], [], []







    def _process_molecule_with_timeout(self, smiles: str, comp_id: str, confidence_thresholds: Dict[str, float], timeout: int = 120) -> Dict:
        """Process a single molecule with timeout protection."""
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.predict_molecule_with_confidence, smiles, confidence_thresholds)
                result = future.result(timeout=timeout)
                result['COMPOUND_ID'] = comp_id
            return result
        except concurrent.futures.TimeoutError:
            self.logger.warning(f"Processing {smiles} timed out after {timeout}s")
            return {
                'COMPOUND_ID': comp_id,
                'SMILES': smiles,
                'xai_status': 'Error - timeout'
            }
        except Exception as e:
            self.logger.error(f"Error processing {smiles}: {str(e)}")
            return {
                'COMPOUND_ID': comp_id,
                'SMILES': smiles,
                'error': str(e)
            }




    def _process_substructure_batch(self, batch_smiles: List[str], batch_ids: List[str], prediction_threshold: float) -> List[Dict]:
        """Process a batch with improved scaffold substructure analysis."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_smiles = {}
            
            for smiles, comp_id in zip(batch_smiles, batch_ids):
                # First get the prediction
                pred_future = executor.submit(self.predict_molecule, smiles)
                future_to_smiles[pred_future] = (smiles, comp_id, "prediction")
                
                # Then analyze scaffolds (optional inventory)
                if getattr(self.config, 'RUN_SCAFFOLD_INVENTORY', False):
                    scaffold_future = executor.submit(self.analyze_molecule_scaffolds, smiles)
                    future_to_smiles[scaffold_future] = (smiles, comp_id, "scaffold")
            
            # Process all results
            predictions = {}
            scaffolds = {}
            
            for future in concurrent.futures.as_completed(future_to_smiles):
                smiles, comp_id, task_type = future_to_smiles[future]
                try:
                    result = future.result()
                    
                    if task_type == "prediction":
                        predictions[(smiles, comp_id)] = result
                    else:  # scaffold
                        scaffolds[(smiles, comp_id)] = result
                except Exception as e:
                    self.logger.error(f"Error processing {smiles} ({task_type}): {str(e)}")
                    if task_type == "prediction":
                        predictions[(smiles, comp_id)] = {
                            'COMPOUND_ID': comp_id, 
                            'SMILES': smiles,
                            'error': str(e)
                        }
                    else:
                        scaffolds[(smiles, comp_id)] = {
                            'error': str(e), 
                            'SMILES': smiles
                        }
        
            # Combine predictions with scaffold analysis if available
            for (smiles, comp_id) in predictions:
                combined_result = predictions[(smiles, comp_id)].copy()
                combined_result['COMPOUND_ID'] = comp_id
            
            # Add classification prediction if threshold is provided
            if prediction_threshold is not None and self.config.classification:
                if 'ensemble_prediction' in combined_result:
                    combined_result['prediction'] = 1 if combined_result['ensemble_prediction'] >= prediction_threshold else 0
                    combined_result.pop('individual_predictions', None)
            
            # Add scaffold information
            if getattr(self.config, 'RUN_SCAFFOLD_INVENTORY', False) and (smiles, comp_id) in scaffolds:
                scaffold_result = scaffolds[(smiles, comp_id)]
                for key, value in scaffold_result.items():
                    if key not in ['SMILES', 'error']:  # Avoid duplicating these keys
                        combined_result[key] = value
            
            results.append(combined_result)
        
        return results




    # Fixed calculate_attribution method for regression
    def calculate_attribution(
        self,
        model,
        smiles: str,
        substructure,
        device: str
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Calculate attribution differences for regression models.
        
        Args:
            model: The trained PyTorch model
            smiles: SMILES string
            substructure: List of atom indices to mask OR list of lists for batch processing
            device: Device to perform computation on
            
        Returns:
            Tuple of (original_predictions, attributions)
        """
        if model is None:
            return None, {}
        
        model.eval()
        try:
            # Handle batch processing case
            if isinstance(substructure, list) and len(substructure) > 0 and isinstance(substructure[0], list):
                return self.batch_calculate_attributions(model, smiles, substructure, device)
            
            # Single substructure case
            original_graph = self._get_graph(smiles, mask=[])
            masked_graph = self._get_graph(smiles, mask=substructure)
            
            if original_graph is None or masked_graph is None:
                self.logger.error(f"Could not construct graphs for SMILES: {smiles}")
                return None, None
            
            with torch.no_grad():
                # Move graphs to correct device
                original_batch = Batch.from_data_list([original_graph]).to(device)
                masked_batch = Batch.from_data_list([masked_graph]).to(device)
                
                original_pred, _ = model(original_batch)
                masked_pred, _ = model(masked_batch)
                
                # Handle multi-property regression output
                if isinstance(original_pred, dict):
                    orig_dict = {}
                    diff_dict = {}
                    
                    for prop in PROPERTY_NAMES:
                        if prop in original_pred and prop in masked_pred:
                            orig_val = float(original_pred[prop].cpu().item())
                            mask_val = float(masked_pred[prop].cpu().item())
                            orig_dict[prop] = orig_val
                            diff_dict[prop] = orig_val - mask_val
                        else:
                            orig_dict[prop] = 0.0
                            diff_dict[prop] = 0.0
                    
                    return orig_dict, diff_dict
                else:
                    # Single output case - convert to property format
                    original_val = float(original_pred.item())
                    masked_val = float(masked_pred.item())
                    diff_val = original_val - masked_val
                    
                    # Assign to first property
                    orig_dict = {PROPERTY_NAMES[0]: original_val}
                    diff_dict = {PROPERTY_NAMES[0]: diff_val}
                    
                    # Fill remaining properties with zeros
                    for prop in PROPERTY_NAMES[1:]:
                        orig_dict[prop] = 0.0
                        diff_dict[prop] = 0.0
                        
                    return orig_dict, diff_dict
                        
        except Exception as e:
            self.logger.error(f"Error in attribution calculation: {str(e)}")
            self.logger.error(traceback.format_exc())
        return None, None

    def batch_calculate_attributions(
        self,
        model,
        smiles: str,
        substructures: List[List[int]],
        device: str
    ) -> Tuple[Dict[str, float], Dict[Tuple[int, ...], Dict[str, float]]]:
        """
        Calculate attributions for multiple substructures efficiently.
        
        Returns:
            Tuple of (original_predictions, {substructure_tuple: {property: attribution}})
        """
        model.eval()
        try:
            # Construct original graph once
            original_graph = self._get_graph(smiles, mask=[])
            if original_graph is None:
                return None, {}
                
            with torch.no_grad():
                original_batch = Batch.from_data_list([original_graph]).to(device)
                original_pred, _ = model(original_batch)
                
                # Create masked graphs for all substructures
                masked_graphs = []
                valid_indices = []
                
                for i, atoms in enumerate(substructures):
                    try:
                        masked_graph = self._get_graph(smiles, mask=atoms)
                        if masked_graph is not None:
                            masked_graphs.append(masked_graph)
                            valid_indices.append(i)
                    except Exception as e:
                        self.logger.debug(f"Error constructing graph for substructure {i}: {str(e)}")
                        continue
                
                if not masked_graphs:
                    if isinstance(original_pred, dict):
                        orig_val = {p: float(original_pred[p].cpu().item()) for p in PROPERTY_NAMES if p in original_pred}
                    else:
                        orig_val = {'value': float(original_pred.item())}
                    return orig_val, {}
                
                attributions = {}
                
                # Process in smaller batches to avoid OOM
                batch_size = 5
                for i in range(0, len(masked_graphs), batch_size):
                    batch_graphs = masked_graphs[i:i+batch_size]
                    batch_indices = valid_indices[i:i+batch_size]
                    
                    # Create batch of graphs
                    masked_batch = Batch.from_data_list([g.to(device) for g in batch_graphs])
                    
                    # Get predictions in a single forward pass
                    masked_preds, _ = model(masked_batch)
                    
                    # Process attributions
                    for j, orig_idx in enumerate(batch_indices):
                        substructure = tuple(substructures[orig_idx])
                        
                        if isinstance(original_pred, dict):
                            diff = {}
                            for p in PROPERTY_NAMES:
                                if p in original_pred:
                                    if isinstance(masked_preds, dict):
                                        # Handle batched dict output
                                        orig_val = float(original_pred[p].cpu().item())
                                        mask_val = float(masked_preds[p][j].cpu().item())
                                        diff[p] = orig_val - mask_val
                                    else:
                                        # Fallback for unexpected output format
                                        diff[p] = 0.0
                            attributions[substructure] = diff
                        else:
                            # Single output case
                            diff_val = float(original_pred.item() - masked_preds[j].item())
                            attributions[substructure] = {'value': diff_val}
                
                # Return original predictions and attributions
                if isinstance(original_pred, dict):
                    # Keep only PRIMARY_TARGET in orig for downstream consumers when focusing FFV
                    try:
                        tgt = getattr(self.config, 'PRIMARY_TARGET', 'FFV')
                        if tgt in original_pred:
                            return {tgt: float(original_pred[tgt].cpu().item())}, attributions
                    except Exception:
                        pass
                    orig = {p: float(original_pred[p].cpu().item()) for p in PROPERTY_NAMES if p in original_pred}
                    return orig, attributions
                else:
                    return {'value': float(original_pred.item())}, attributions
                    
        except Exception as e:
            self.logger.error(f"Error in batch attribution calculation: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, {}






    def process_dataset(
        self,
        input_csv: str,
        output_csv: str,
        batch_size: int,
        substructure_type: str,
        start_idx: int = 0,
        end_idx: int = None,
        resume: bool = False,
        num_workers: int = 4,
        device: str = 'cpu'
    ):
        """Process a specific subset of the dataset with optimized batch processing and confidence-based XAI."""
        
        try:
            if input_csv != self.input_csv:
                raise ValueError("Input file mismatch. Please initialize a new PredictionManager.")

            # Handle None or "None" values consistently 
            if substructure_type == "None":
                substructure_type = None
                
            # Explicitly set substructure_type on config
            if substructure_type is not None:
                self.config.substructure_type = substructure_type
                self.logger.info(f"Substructure analysis type set to: {self.config.substructure_type}")
            else:
                self.config.substructure_type = None
                self.logger.warning("No substructure analysis will be performed (substructure_type is None)")

            # Initialize caches
            if not hasattr(self, '_substructure_cache'):
                self._substructure_cache = {}
            if not hasattr(self, '_analysis_cache'):
                self._analysis_cache = {}

            # Start timing
            global_start_time = time.time()

            # Read full file to get total count
            try:
                if input_csv.lower().endswith('.parquet'):
                    df_full = pd.read_parquet(input_csv)
                else:
                    df_full = pd.read_csv(input_csv)
                df_full.columns = df_full.columns.str.strip()
            except Exception as e:
                raise ValueError(f"Error reading input file {input_csv}: {str(e)}")

            self.logger.info(f"Available columns: {df_full.columns.tolist()}")

            # Validate required columns
            required_columns = ['SMILES', 'COMPOUND_ID']
            missing_columns = [col for col in required_columns if col not in df_full.columns]
            if missing_columns:
                raise ValueError(f"Required columns {missing_columns} not found. Available columns: {df_full.columns.tolist()}")

            # Clean and validate data
            empty_smiles = df_full['SMILES'].isna().sum()
            empty_ids = df_full['COMPOUND_ID'].isna().sum()
            if empty_smiles > 0 or empty_ids > 0:
                self.logger.warning(f"Found {empty_smiles} empty SMILES and {empty_ids} empty COMPOUND_IDs")

            df_full = df_full.dropna(subset=['SMILES', 'COMPOUND_ID'])
            
            full_dataset_size = len(df_full)
            if full_dataset_size == 0:
                raise ValueError("No valid molecules found in input file after cleaning")
            
            # Determine subset range
            if end_idx is None or end_idx >= full_dataset_size:
                end_idx = full_dataset_size - 1
                
            if start_idx < 0:
                start_idx = 0
                
            # Extract only our subset for processing
            df = df_full.iloc[start_idx:end_idx+1].reset_index(drop=True)
            total_molecules = len(df)

            self.logger.info(f"Processing subset of {total_molecules} molecules (indices {start_idx} to {end_idx}) out of {full_dataset_size} total")

            # Representation selection for each row: add text_repr and used_bigsmiles; optionally filter
            mode = str(getattr(self.config, 'PREDICT_REPR', 'auto') or 'auto').lower()
            require_bigs = bool(getattr(self.config, 'BIGSMILES_REQUIRED', False))
            if mode not in ('auto', 'smiles', 'bigsmiles'):
                self.logger.warning(f"Unknown PREDICT_REPR={mode}; defaulting to 'auto'")
                mode = 'auto'
            # Prepare columns
            text_repr_list: List[str] = []
            used_bigs_list: List[bool] = []
            canon_smiles_list: List[str] = []
            keep_mask = []
            local_total = 0
            local_ok = 0
            repr_map_local: Dict[str, Dict[str, Any]] = {}
            for idx, row in df.iterrows():
                s = str(row['SMILES'])
                comp_id = row['COMPOUND_ID']
                try:
                    sm2, txt, ok = self._choose_text_repr(s, mode=mode, require_bigs=require_bigs)
                except Exception as e:
                    if require_bigs:
                        # skip with warning
                        self.logger.warning(f"Skipping {comp_id}: BigSMILES required but conversion failed: {e}")
                        keep_mask.append(False)
                        continue
                    # fallback
                    sm2, txt, ok = s, s, False
                local_total += 1
                if ok:
                    local_ok += 1
                keep_mask.append(True)
                canon_smiles_list.append(sm2)
                text_repr_list.append(txt)
                used_bigs_list.append(bool(ok))
                # Populate map for downstream JSON emission and result augmentation
                repr_map_local[str(comp_id)] = {
                    'canon_smiles': sm2,
                    'text_repr': txt,
                    'used_bigsmiles': bool(ok),
                }
            # Apply filter if any rows were skipped
            if not all(keep_mask):
                df = df[keep_mask].reset_index(drop=True)
                # Align the prepared lists to filtered df length
                # They already only include kept rows because we skipped appending on skipped ones
            # Attach prepared columns
            if len(df) == len(text_repr_list):
                df['SMILES'] = canon_smiles_list
                df['text_repr'] = text_repr_list
                df['used_bigsmiles'] = used_bigs_list
            else:
                # Defensive: recompute lists aligned to df
                aligned_canon = []
                aligned_text = []
                aligned_used = []
                for _, row in df.iterrows():
                    m = repr_map_local.get(str(row['COMPOUND_ID']))
                    if m is None:
                        sm2, txt, ok = self._choose_text_repr(str(row['SMILES']), mode=mode, require_bigs=False)
                        m = {'canon_smiles': sm2, 'text_repr': txt, 'used_bigsmiles': bool(ok)}
                    aligned_canon.append(m['canon_smiles'])
                    aligned_text.append(m['text_repr'])
                    aligned_used.append(bool(m['used_bigsmiles']))
                df['SMILES'] = aligned_canon
                df['text_repr'] = aligned_text
                df['used_bigsmiles'] = aligned_used
            # Update global counters and map
            self._bigs_ok += local_ok
            self._bigs_total += local_total
            self._repr_map.update(repr_map_local)
            # Log coverage for this subset
            try:
                pct = (local_ok / local_total * 100.0) if local_total else 0.0
                self.logger.info(f"BigSMILES coverage (subset): {local_ok}/{local_total} ({pct:.1f}%)")
            except Exception:
                pass
            # Update total_molecules in case rows were skipped due to BIGSMILES_REQUIRED
            total_molecules = len(df)
            
            # Use a shard-specific checkpoint for better isolation
            shard_id = ""
            if output_csv:
                # Try to extract shard ID from output filename
                filename = os.path.basename(output_csv)
                if "shard" in filename:
                    shard_id = "_shard" + filename.split("shard")[-1].split(".")[0]
                    
            # Create shard-specific checkpoint path
            checkpoint_path = self.checkpoint_file
            if shard_id:
                checkpoint_dir = os.path.dirname(self.checkpoint_file)
                checkpoint_base = os.path.basename(self.checkpoint_file)
                checkpoint_name, checkpoint_ext = os.path.splitext(checkpoint_base)
                checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}{shard_id}{checkpoint_ext}")
                
            self.logger.info(f"Using checkpoint file: {checkpoint_path}")
            
            # Load checkpoint data if resuming
            all_results = []
            current_idx = 0
            
            if resume and os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                        
                    # Extract only relevant results for our range
                    saved_results = checkpoint.get('results', [])
                    if saved_results:
                        # Only include results that match our molecule IDs
                        subset_ids = set(df['COMPOUND_ID'].tolist())
                        all_results = [r for r in saved_results if r.get('COMPOUND_ID') in subset_ids]
                        
                        # Determine how many we've already processed
                        current_idx = len(all_results)
                        
                        self.logger.info(f"Resuming from checkpoint with {current_idx} molecules already processed")
                except Exception as e:
                    self.logger.error(f"Error loading checkpoint: {str(e)}")
                    self.logger.warning("Starting from beginning")
                    current_idx = 0
                    all_results = []
            else:
                self.logger.info("Starting fresh processing")
                current_idx = 0
                all_results = []

            # Determine confidence thresholds using a sample of the dataset
            self.logger.info("Analyzing dataset confidence...")
            confidence_thresholds = self.analyze_dataset_confidence(input_csv, sample_size=1000)
            self.logger.info(f"Confidence thresholds determined: {confidence_thresholds}")

            # Store for batch processing
            self.confidence_thresholds = confidence_thresholds

            # Save initial checkpoint
            self.save_checkpoint({
                'current_idx': current_idx,
                'results': all_results,
                'total_molecules': total_molecules,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'confidence_thresholds': confidence_thresholds
            }, checkpoint_path)

            # Processing time tracking
            start_time = time.time()
            last_save_time = start_time
            
            # Clean memory before starting
            try:
                self._clean_memory()
            except AttributeError:
                self.logger.warning("_clean_memory method not found, using fallback memory cleanup")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Calculate remaining molecules to process
            remaining_molecules = total_molecules - current_idx
            if remaining_molecules <= 0:
                self.logger.info(f"All {total_molecules} molecules in this range already processed")
                # Still save final results even if nothing to process
                if all_results:
                    self._save_final_results(all_results, output_csv, substructure_type)
                return
            else:
                self.logger.info(f"Processing {remaining_molecules} remaining molecules")
            
            # Determine batch count for logging
            total_batches = (remaining_molecules + batch_size - 1) // batch_size

            # Main processing loop
            subset_idx = current_idx
            batch_num = 1
            
            while subset_idx < total_molecules and not self.should_terminate:
                # Check for termination signal
                if self.should_terminate:
                    self.logger.info("Termination signal received, saving current progress...")
                    self.save_checkpoint({
                        'current_idx': subset_idx,
                        'results': all_results,
                        'total_molecules': total_molecules,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'confidence_thresholds': confidence_thresholds
                    }, checkpoint_path)
                    break

                # Get current batch from our subset
                batch_start_time = time.time()
                batch_end_idx = min(subset_idx + batch_size, total_molecules)
                batch_df = df.iloc[subset_idx:batch_end_idx]
                batch_smiles = batch_df['SMILES'].tolist()
                batch_ids = batch_df['COMPOUND_ID'].tolist()
                batch_text = batch_df['text_repr'].tolist() if 'text_repr' in batch_df.columns else [''] * len(batch_ids)
                batch_used = batch_df['used_bigsmiles'].tolist() if 'used_bigsmiles' in batch_df.columns else [False] * len(batch_ids)
                
                batch_size_actual = len(batch_smiles)

                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({batch_size_actual} molecules)")
                
                # Log GPU memory before processing
                if torch.cuda.is_available():
                    try:
                        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                        gpu_max = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        self.logger.info(f"GPU memory: current {gpu_mem:.2f} GB, peak {gpu_max:.2f} GB")
                    except Exception as e:
                        self.logger.warning(f"Could not log GPU memory: {str(e)}")
                
                # Process the batch with confidence thresholds
                batch_results = []
                try:
                    batch_results = self.process_batch(
                        batch_smiles, 
                        batch_ids,
                        confidence_thresholds  # Pass confidence thresholds
                    )
                    # Augment results with representation fields and track coverage per batch
                    b_ok = 0
                    for i, res in enumerate(batch_results):
                        try:
                            cid = res.get('COMPOUND_ID', batch_ids[i])
                            # Prefer map to ensure consistency
                            m = self._repr_map.get(str(cid))
                            if m is None and i < len(batch_text):
                                m = {'text_repr': batch_text[i], 'used_bigsmiles': bool(batch_used[i])}
                            if m is not None:
                                res['text_repr'] = m.get('text_repr', '')
                                res['used_bigsmiles'] = bool(m.get('used_bigsmiles', False))
                                if res['used_bigsmiles']:
                                    b_ok += 1
                        except Exception:
                            continue
                    try:
                        pct_b = (b_ok / max(1, len(batch_results))) * 100.0
                        self.logger.info(f"BigSMILES coverage (batch): {b_ok}/{len(batch_results)} ({pct_b:.1f}%)")
                    except Exception:
                        pass
                    
                    # Verify we got results
                    if not batch_results:
                        self.logger.warning(f"Batch {batch_num} returned no results! Trying one by one...")
                        # Fallback to one-by-one processing with confidence
                        for smiles, comp_id in zip(batch_smiles, batch_ids):
                            try:
                                result = self.predict_molecule_with_confidence(smiles, confidence_thresholds)
                                result['COMPOUND_ID'] = comp_id
                                batch_results.append(result)
                            except Exception as e:
                                self.logger.error(f"Error processing {smiles}: {str(e)}")
                                batch_results.append({
                                    'COMPOUND_ID': comp_id,
                                    'SMILES': smiles,
                                    'error': str(e),
                                    'xai_status': f'Error - {str(e)}'
                                })
                except Exception as e:
                    self.logger.error(f"Error processing batch: {str(e)}")
                    self.logger.debug(traceback.format_exc())
                    # Try individual molecules as a fallback
                    self.logger.info("Attempting to process molecules individually...")
                    for smiles, comp_id in zip(batch_smiles, batch_ids):
                        try:
                            result = self.predict_molecule_with_confidence(smiles, confidence_thresholds)
                            result['COMPOUND_ID'] = comp_id
                            batch_results.append(result)
                        except Exception as mol_e:
                            self.logger.error(f"Error processing {smiles}: {str(mol_e)}")
                            batch_results.append({
                                'COMPOUND_ID': comp_id,
                                'SMILES': smiles,
                                'error': str(mol_e),
                                'xai_status': f'Error - {str(mol_e)}'
                            })

                # Update results and save progress
                if batch_results:
                    self.logger.info(f"Adding {len(batch_results)} results from batch {batch_num}")
                    all_results.extend(batch_results)
                    
                    # Log XAI status summary for this batch
                    xai_completed = sum(1 for r in batch_results if r.get('xai_status') == 'Completed')
                    xai_skipped = sum(1 for r in batch_results if 'Skipped' in r.get('xai_status', ''))
                    xai_error = sum(1 for r in batch_results if 'Error' in r.get('xai_status', ''))
                    
                    self.logger.info(f"Batch {batch_num} XAI summary: {xai_completed} completed, {xai_skipped} skipped, {xai_error} errors")
                else:
                    # Controller update and batch status
                    try:
                        prop = getattr(self.config, 'PRIMARY_TARGET', 'FFV')
                        tau = float(getattr(self.config, 'TAU_STD', 0.05))
                        delta_min = float(getattr(self.config, 'DELTA_MIN', 0.01))
                        est_flags = []
                        used_bigs = 0
                        for rr in batch_results:
                            predv = rr.get(f'pred_{prop}')
                            stdv = rr.get(f'{prop}_std')
                            confv = rr.get(f'{prop}_confidence')
                            if rr.get('used_bigsmiles'): used_bigs += 1
                            hvy = (predv is not None and confv == 'high' and stdv is not None and float(stdv) <= tau and abs(float(predv) - float(self.mu_ffv)) >= delta_min)
                            est_flags.append(bool(hvy))
                        avg_sec = (time.time() - batch_start_time) / max(1, len(batch_results))
                        self._controller_update(est_flags, avg_secs_per_mol=avg_sec)
                        rate_win = (sum(self._heavy_window)/len(self._heavy_window)) if len(self._heavy_window)>0 else 0.0
                        kept_scaf = getattr(self, '_last_batch_kept_scaf', 0.0)
                        kept_subs = getattr(self, '_last_batch_kept_subs', 0.0)
                        pct_bigs = 100.0 * used_bigs / max(1, len(batch_results))
                        self.logger.info(
                            f"XAI batch: kept_scaf=avg {kept_scaf:.1f}, kept_subs=avg {kept_subs:.1f}, "
                            f"heavy_rate(win)={rate_win:.2f}, tau={self.config.TAU_STD:.3f}, "
                            f"delta={self.config.DELTA_MIN:.3f}, avg={avg_sec:.1f}s/mol, bigs={pct_bigs:.1f}%")
                    except Exception as _e:
                        self.logger.debug(f'Controller/log status skipped: {_e}')
                        self.logger.warning(f"Batch {batch_num} produced no results!")
                        # Add empty results to avoid getting stuck
                        for smiles, comp_id in zip(batch_smiles, batch_ids):
                            all_results.append({
                                'COMPOUND_ID': comp_id, 
                                'SMILES': smiles,
                                'error': 'Failed batch processing with no specific error',
                                'xai_status': 'Error - batch processing failed'
                            })
                
                # Calculate and log timing information
                current_time = time.time()
                batch_time = current_time - batch_start_time
                elapsed_time = current_time - start_time
                molecules_processed = subset_idx + len(batch_results)
                
                avg_time_per_mol = batch_time / batch_size_actual
                overall_avg_time = elapsed_time / molecules_processed if molecules_processed > 0 else 0
                
                remaining_mols = total_molecules - molecules_processed
                estimated_time = remaining_mols * overall_avg_time if remaining_mols > 0 else 0
                
                self.logger.info(
                    f"Batch {batch_num} completed in {self.format_time(batch_time)} - "
                    f"{avg_time_per_mol:.2f}s per molecule"
                )
                
                self.logger.info(
                    f"Progress: {molecules_processed}/{total_molecules} molecules "
                    f"({(molecules_processed / total_molecules) * 100:.1f}%)"
                )
                
                self.logger.info(
                    f"Overall average: {overall_avg_time:.2f}s per molecule - "
                    f"Estimated time remaining: {self.format_time(estimated_time)}"
                )
                
                # Save checkpoint and temporary results periodically
                if (current_time - last_save_time) >= 300 or batch_end_idx == total_molecules:
                    self.save_checkpoint({
                        'current_idx': subset_idx + len(batch_results),
                        'results': all_results,
                        'total_molecules': total_molecules,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'confidence_thresholds': confidence_thresholds
                    }, checkpoint_path)
                    
                    # Save partial results
                    temp_file = f"{output_csv}.temp"
                    try:
                        pd.DataFrame(all_results).to_csv(temp_file, index=False)
                        self.logger.info(f"Saved checkpoint and temporary results to {temp_file}")
                    except Exception as save_e:
                        self.logger.warning(f"Could not save temporary results: {str(save_e)}")
                    last_save_time = current_time
                
                # Clean memory after batch
                try:
                    self._clean_memory()
                except AttributeError:
                    self.logger.debug("Using fallback memory cleanup after batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
                # Update for next iteration
                subset_idx = batch_end_idx
                batch_num += 1

            # Save final results with proper formatting
            if all_results:
                # Log number of results before saving
                self.logger.info(f"Saving {len(all_results)} total results")
                # Final BigSMILES coverage over the run
                try:
                    pct_final = (self._bigs_ok / self._bigs_total * 100.0) if self._bigs_total else 0.0
                    self.logger.info(f"BigSMILES coverage (total): {self._bigs_ok}/{self._bigs_total} ({pct_final:.1f}%)")
                except Exception:
                    pass
                
                # Log final XAI summary
                total_completed = sum(1 for r in all_results if r.get('xai_status') == 'Completed')
                total_skipped = sum(1 for r in all_results if 'Skipped' in r.get('xai_status', ''))
                total_errors = sum(1 for r in all_results if 'Error' in r.get('xai_status', ''))
                
                self.logger.info(f"Final XAI summary: {total_completed} completed, {total_skipped} skipped, {total_errors} errors")
                
                # Check for substructure data in results
                if hasattr(self.config, 'substructure_type') and self.config.substructure_type:
                    pattern = f"{self.config.substructure_type}_substructure"
                    has_substructure = any(any(pattern in key for key in result.keys()) for result in all_results if isinstance(result, dict))
                    if has_substructure:
                        self.logger.info(f"Results contain {self.config.substructure_type} substructure data")
                    else:
                        self.logger.warning(f"No {self.config.substructure_type} substructure data found in results!")
                
                self._save_final_results(all_results, output_csv, substructure_type)
                
                # Log overall statistics
                total_time = time.time() - global_start_time
                overall_speed = len(all_results) / total_time if total_time > 0 else 0
                
                self.logger.info("=" * 50)
                self.logger.info(f"Processing completed in {self.format_time(total_time)}")
                self.logger.info(f"Average processing speed: {overall_speed:.2f} molecules/second")
                self.logger.info(f"Final results saved to: {output_csv}")
                self.logger.info("=" * 50)
            else:
                self.logger.warning("No results to save!")
                
        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Try to save progress even on error
            if locals().get('all_results') and locals().get('current_idx'):
                error_output = f"{output_csv}.error_partial"
                try:
                    pd.DataFrame(all_results).to_csv(error_output, index=False)
                    self.logger.info(f"Saved partial results to {error_output}")
                except:
                    self.logger.error("Failed to save partial results")
            raise
                



    




    def _process_simple_batch(self, batch_smiles: List[str], batch_ids: List[str]) -> List[Dict]:
        """
        Process a batch without substructure analysis - for basic polymer property prediction only.
        
        Args:
            batch_smiles: List of SMILES strings
            batch_ids: List of compound IDs
            
        Returns:
            List of prediction results
        """
        batch_results = []
        
        for smiles, comp_id in zip(batch_smiles, batch_ids):
            try:
                # Get basic polymer property predictions
                result = self.predict_molecule(smiles)
                # Apply FFV calibration if available (additive fields)
                try:
                    if getattr(self.config, 'CALIBRATE_FFV', False) and self.ffv_calibrator is not None and 'pred_FFV' in result:
                        cal = self.ffv_calibrator.predict([float(result['pred_FFV'])])[0]
                        result['pred_FFV_calibrated'] = float(cal)
                except Exception:
                    pass
                result['COMPOUND_ID'] = comp_id
                
                # Log successful prediction
                self.logger.debug(f"Successfully predicted properties for {comp_id}")
                
                batch_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing compound ID {comp_id} (SMILES: {smiles}): {str(e)}")
                self.logger.debug(traceback.format_exc())
                
                # Add error result to maintain batch consistency
                batch_results.append({
                    'COMPOUND_ID': comp_id,
                    'SMILES': smiles,
                    'error': str(e),
                    'n_models': 0
                })
        
        return batch_results

    def _process_confidence_batch(self, batch_smiles: List[str], batch_ids: List[str], 
                                confidence_thresholds: Dict[str, float]) -> List[Dict]:
        """
        Process a batch with confidence-based XAI analysis.
        
        Args:
            batch_smiles: List of SMILES strings
            batch_ids: List of compound IDs
            confidence_thresholds: Confidence thresholds for each property
            
        Returns:
            List of prediction results with confidence-based XAI
        """
        batch_results = []
        
        for smiles, comp_id in zip(batch_smiles, batch_ids):
            try:
                # Get predictions with confidence analysis
                result = self.predict_molecule_with_confidence(smiles, confidence_thresholds)
                # Apply FFV calibration if available (additive fields)
                try:
                    if self.ffv_calibrator is not None and 'pred_FFV' in result:
                        cal = self.ffv_calibrator.predict([float(result['pred_FFV'])])[0]
                        result['pred_FFV_calibrated'] = float(cal)
                except Exception:
                    pass
                result['COMPOUND_ID'] = comp_id
                
                # Log successful prediction
                high_conf_props = [prop for prop in PROPERTY_NAMES 
                                if result.get(f"{prop}_confidence") == "high"]
                self.logger.debug(f"Predicted {comp_id} - High confidence: {high_conf_props}")
                
                batch_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing compound ID {comp_id} (SMILES: {smiles}): {str(e)}")
                self.logger.debug(traceback.format_exc())
                
                # Add error result
                batch_results.append({
                    'COMPOUND_ID': comp_id,
                    'SMILES': smiles,
                    'error': str(e)
                })
        
        return batch_results

    def _process_full_xai_batch(self, batch_smiles: List[str], batch_ids: List[str]) -> List[Dict]:
        """
        Process a batch with full XAI substructure analysis for all molecules.
        
        Args:
            batch_smiles: List of SMILES strings
            batch_ids: List of compound IDs
            
        Returns:
            List of prediction results with full XAI analysis
        """
        batch_results = []
        
        for smiles, comp_id in zip(batch_smiles, batch_ids):
            try:
                # Get basic predictions first
                result = self.predict_molecule(smiles)
                # Apply FFV calibration if available (additive fields)
                try:
                    if self.ffv_calibrator is not None and 'pred_FFV' in result:
                        cal = self.ffv_calibrator.predict([float(result['pred_FFV'])])[0]
                        result['pred_FFV_calibrated'] = float(cal)
                except Exception:
                    pass
                result['COMPOUND_ID'] = comp_id
                
                # Add full substructure analysis if requested
                if hasattr(self.config, 'substructure_type') and self.config.substructure_type:
                    try:
                        sc_attrs, sc_smis, sc_subs = self.analyze_substructures(
                            smiles, self.config.substructure_type
                        )
                        
                        # Add scaffold information to result
                        t = self.config.substructure_type
                        for i, (attr, smi) in enumerate(zip(sc_attrs, sc_smis)):
                            result[f'{t}_substructure_{i}_attribution'] = attr
                            result[f'{t}_substructure_{i}_smiles'] = smi
                            
                            # Add substituent information
                            if i < len(sc_subs):
                                for j, sub in enumerate(sc_subs[i]):
                                    result[f'{t}_substituent_{i}_{j}_smiles'] = sub.get('smiles', '')
                                    result[f'{t}_substituent_{i}_{j}_context'] = sub.get('context', '')
                                    result[f'{t}_substituent_{i}_{j}_attribution'] = sub.get('attribution', 0.0)
                        
                        self.logger.debug(f"Added XAI analysis for {comp_id} - {len(sc_attrs)} scaffolds found")
                        
                    except Exception as xai_e:
                        self.logger.warning(f"XAI analysis failed for {comp_id}: {str(xai_e)}")
                        result['xai_error'] = str(xai_e)
                
                batch_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing compound ID {comp_id} (SMILES: {smiles}): {str(e)}")
                self.logger.debug(traceback.format_exc())
                
                # Add error result
                batch_results.append({
                    'COMPOUND_ID': comp_id,
                    'SMILES': smiles,
                    'error': str(e)
                })
        
        return batch_results

    # Updated process_batch method that chooses the appropriate processing strategy
    def process_batch(self, smiles_list: List[str], compound_ids: List[str], confidence_thresholds: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Process a batch of molecules with the appropriate strategy based on configuration.
        
        Args:
            smiles_list: List of SMILES strings
            compound_ids: List of compound IDs
            confidence_thresholds: Confidence thresholds for XAI triggering
            
        Returns:
            List of prediction results
        """
        batch_start_time = time.time()
        
        # Clean memory before starting batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Determine processing strategy based on configuration
        if not hasattr(self.config, 'substructure_type') or not self.config.substructure_type:
            # Simple prediction only - fastest (no XAI requested)
            self.logger.debug("Using simple batch processing (no XAI - substructure_type is None)")
            results = self._process_simple_batch(smiles_list, compound_ids)
            
        elif confidence_thresholds is not None:
            # Confidence-based XAI - this is the main path we want
            self.logger.debug(f"Using confidence-based batch processing with thresholds: {confidence_thresholds}")
            results = self._process_confidence_batch(smiles_list, compound_ids, confidence_thresholds)
            
        else:
            # This should rarely happen - fallback to full XAI
            self.logger.warning("No confidence thresholds provided but XAI requested - using full XAI")
            results = self._process_full_xai_batch(smiles_list, compound_ids)
        
        # Emit XAI JSONs if enabled
        try:
            if getattr(self.config, 'EMIT_XAI_JSON', True):
                K = getattr(self.config, 'XAI_TOPK', 10)
                heavy_flags = []
                for r in results:
                    comp_id = r.get('COMPOUND_ID', 'unknown')
                    smiles = r.get('SMILES', '')
                    per_target = {}
                    ensemble_std = {}
                    notes: List[str] = []

                    # Prepare molecule once for fragment SMILES
                    mol = Chem.MolFromSmiles(smiles)

                    # FFV-only XAI
                    prop = getattr(self.config, 'PRIMARY_TARGET', 'FFV')
                    pred = r.get(f'pred_{prop}')
                    std = r.get(f'{prop}_std')
                    if pred is not None:
                        # Gate heavy Δ-masking by confidence
                        conf = r.get(f'{prop}_confidence', 'unknown')
                        # Adaptive gating for heavy 94-masking
                        ffv_std = std
                        tau = float(getattr(self.config, 'TAU_STD', 0.05))
                        delta_min = float(getattr(self.config, 'DELTA_MIN', 0.01))
                        margin = abs(float(pred) - float(self.mu_ffv)) if self.mu_ffv is not None else float('inf')

                        # Check if we should compute XAI for all molecules (bypass gating)
                        compute_for_all = getattr(self.config, 'COMPUTE_XAI_FOR_ALL', False)
                        if compute_for_all:
                            # Bypass gating - compute XAI for all molecules
                            do_frag = True
                            if not hasattr(self, '_logged_xai_all_mode'):
                                self.logger.info("COMPUTE_XAI_FOR_ALL=True: Computing XAI for all molecules (gating disabled)")
                                self._logged_xai_all_mode = True
                        else:
                            # Original gating logic
                            do_frag = (conf == 'high') and (ffv_std is not None and float(ffv_std) <= tau) and (margin >= delta_min)

                        frag_entries: List[Dict[str, Any]] = []
                        if do_frag:
                            try:
                                candidates = self._collect_candidate_masks(smiles)
                                selected = self._select_non_overlapping_masks(candidates, K)
                                attributed = self._compute_mask_deltas(smiles, selected, prop)
                                # Split by sign and rank by |raw_delta|
                                pos = [d for d in attributed if d['raw_delta'] >= 0]
                                neg = [d for d in attributed if d['raw_delta'] < 0]
                                pos = sorted(pos, key=lambda d: abs(d['raw_delta']), reverse=True)
                                neg = sorted(neg, key=lambda d: abs(d['raw_delta']), reverse=True)
                                k_pos = K // 2
                                k_neg = K - k_pos
                                chosen = pos[:k_pos] + neg[:k_neg]
                                if len(chosen) < K:
                                    leftovers = pos[k_pos:] + neg[k_neg:]
                                    leftovers = sorted(leftovers, key=lambda d: abs(d['raw_delta']), reverse=True)
                                    chosen += leftovers[:(K - len(chosen))]
                                for d in chosen:
                                    frag_smiles = None
                                    if mol is not None:
                                        try:
                                            frag_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=d['atoms'], kekuleSmiles=True)
                                        except Exception:
                                            frag_smiles = None
                                    frag_entries.append({
                                        'mask_id': d['mask_id'],
                                        'mask_family': d['mask_family'],
                                        'atoms': d['atoms'],
                                        'smarts': frag_smiles or '',
                                        'raw_delta': float(d['raw_delta']),
                                        'norm_delta': float(d['norm_delta']),
                                        'size': int(d['size'])
                                    })
                            except Exception as xe:
                                self.logger.debug(f"Fragment ?-masking failed for {prop}: {xe}")
                        else:
                            notes.append('lightweight XAI emitted \(gated\)')

                        # Tokens placeholder only (text model not active)
                        tok_entries = []
                        tok_note = 'text model not active'
                        frag_sets = [set(e['atoms']) for e in frag_entries]
                        top_atoms = set().union(*frag_sets) if frag_sets else set()
                        tok_set = set()
                        jacc = 0.0
                        if len(top_atoms) + len(tok_set) > 0:
                            jacc = float(len(top_atoms & tok_set) / len(top_atoms | tok_set))
                        if not tok_entries:
                            notes.append('token attributions unavailable at inference')

                        per_target[prop] = {
                            'prediction': float(pred),
                            'pi': [float(pred - 1.96 * (std or 0.0)), float(pred + 1.96 * (std or 0.0))],
                            'fragments': frag_entries,
                            'tokens': tok_entries if tok_entries else [{'note': tok_note}],
                            'shap_groups': [],
                            'agreement': {f'frag_vs_tok@{K}': jacc}
                        }
                        ensemble_std[prop] = float(std) if std is not None else None

                    # Indicate BigSMILES mapping unavailable if none of block/connector/end_group in masks
                    try:
                        cands = self._collect_candidate_masks(smiles)
                        has_non_fragment = any(m.get('mask_family') in ('block', 'connector', 'end_group') for m in cands)
                        if not has_non_fragment:
                            notes.append('bigsmiles mapping unavailable at inference')
                    except Exception:
                        notes.append('bigsmiles mapping unavailable at inference')

                    # Summary and guards
                    summary = {
                        'ensemble_std': ensemble_std,
                        'uncertainty_flag': bool((ensemble_std.get('FFV') or 0) > 0)
                    }
                    if 'FFV' not in per_target:
                        per_target['FFV'] = {'note': 'FFV unavailable for this molecule'}
                    else:
                        if notes:
                            per_target['FFV'].setdefault('notes', notes)

                    # Calculate agreement metrics (if enabled)
                    if add_agreement_to_xai_json is not None and 'FFV' in per_target:
                        try:
                            ffv_pred = per_target['FFV'].get('prediction')
                            ffv_std = ensemble_std.get('FFV', 0.0)
                            if ffv_pred is not None and self.mu_ffv is not None:
                                per_target['FFV'] = add_agreement_to_xai_json(
                                    per_target_ffv=per_target['FFV'],
                                    prediction=ffv_pred,
                                    ensemble_std=ffv_std,
                                    baseline_mean=self.mu_ffv,
                                    config=self.config
                                )
                        except Exception as e:
                            self.logger.warning(f"Failed to calculate agreement metrics: {e}")

                    # Write files
                    self._emit_xai_json(comp_id, smiles, per_target, summary)
        except Exception as e:
            self.logger.warning(f'XAI JSON emission failed: {e}')

        # Log batch completion
        batch_time = time.time() - batch_start_time
        self.logger.info(
            f"Batch completed in {self.format_time(batch_time)} - "
            f"Average: {batch_time/len(smiles_list):.2f}s per molecule"
        )
        
        return results
    







    def _save_final_results(self, all_results: List[Dict], output_csv: str, substructure_type: str):
        """Save final results with correct polymer property columns."""
        # Create DataFrame from results
        final_df = pd.DataFrame(all_results)
        
        # Log initial data size
        self.logger.info(f"Initial dataframe has {len(final_df)} rows and {len(final_df.columns)} columns")

        # Remove atom weights to avoid serialization issues
        if 'atom_weights' in final_df.columns:
            self.logger.info("Removing atom_weights column to avoid serialization issues")
            final_df = final_df.drop('atom_weights', axis=1)

        # Base columns for polymer regression
        base = ['COMPOUND_ID', 'SMILES']
        # Add representation columns if present (backward compatible: only add, never remove)
        for extra_col in ['text_repr', 'used_bigsmiles']:
            if extra_col in final_df.columns and extra_col not in base:
                base.append(extra_col)
        for prop in PROPERTY_NAMES:
            base.append(f'pred_{prop}')
        # Add calibrated FFV if present
        if 'pred_FFV_calibrated' in final_df.columns:
            base.append('pred_FFV_calibrated')
        for prop in PROPERTY_NAMES:
            std_col = f'{prop}_std'
            if std_col in final_df.columns:
                base.append(std_col)
            conf_col = f'{prop}_confidence'
            if conf_col in final_df.columns:
                base.append(conf_col)
        base.append('n_models')
        
        # Add strategic WMAE if present
        if 'strategic_wmae_estimate' in final_df.columns:
            base.append('strategic_wmae_estimate')
        
        # Add XAI status column if present
        if 'xai_status' in final_df.columns:
            base.append('xai_status')

        # Parse and sort columns for scaffolds and substituents
        sc_columns = []  # For scaffold columns
        sub_columns = []  # For substituent columns
        
        if substructure_type:
            t = substructure_type
            
            # Extract all scaffold columns
            # Pattern includes property name (e.g., murcko_substructure_0_FFV_attribution)
            scaffold_attr_pattern = re.compile(f'{t}_substructure_(\d+)_(\w+)_attribution')
            scaffold_smiles_pattern = re.compile(f'{t}_substructure_(\d+)_smiles')
            
            # First get all scaffold attribution and smiles columns
            scaffold_attr_cols = []
            scaffold_smiles_cols = []
            
            for col in final_df.columns:
                attr_match = scaffold_attr_pattern.match(col)
                if attr_match:
                    idx = int(attr_match.group(1))  # First group is the index
                    # group(2) would be the property name (FFV, Tg, etc.) if needed
                    scaffold_attr_cols.append((idx, col))
                    continue

                smiles_match = scaffold_smiles_pattern.match(col)
                if smiles_match:
                    idx = int(smiles_match.group(1))
                    scaffold_smiles_cols.append((idx, col))
            
            # Sort scaffold columns by index
            scaffold_attr_cols.sort()
            scaffold_smiles_cols.sort()
            
            # Add to scaffold columns list
            sc_columns = [col for _, col in scaffold_attr_cols] + [col for _, col in scaffold_smiles_cols]
            
            # Extract all substituent columns
            sub_pattern = re.compile(f'{t}_substituent_(\d+)_(\d+)_(smiles|context|attribution)')
            sub_cols_dict = {}
            
            for col in final_df.columns:
                sub_match = sub_pattern.match(col)
                if sub_match:
                    sc_idx = int(sub_match.group(1))
                    sub_idx = int(sub_match.group(2))
                    prop = sub_match.group(3)
                    
                    # Create a composite key for sorting
                    key = (sc_idx, sub_idx, prop)
                    sub_cols_dict[key] = col
            
            # Sort substituent columns first by scaffold index, then by substituent index
            # For each substituent, order as smiles, context, attribution
            prop_order = {'smiles': 0, 'context': 1, 'attribution': 2}
            sorted_sub_keys = sorted(sub_cols_dict.keys(), 
                                    key=lambda k: (k[0], k[1], prop_order.get(k[2], 3)))
            
            sub_columns = [sub_cols_dict[k] for k in sorted_sub_keys]
        
        # Add FFV-only convenience fields to base outputs if present
        for key in ['ffv_pred', 'prediction_std', 'ffv_margin_from_mean', 'xai_method']:
            if key in final_df.columns and key not in base:
                base.append(key)

        # Combine all columns
        ordered_columns = base + sc_columns + sub_columns
        
        # Add error column if present
        if 'error' in final_df.columns:
            ordered_columns.append('error')
        
        # Keep only columns that actually exist in the dataframe
        final_columns = [col for col in ordered_columns if col in final_df.columns]
        
        # Log the columns we found for debugging
        self.logger.info(f"Column types found: {len(final_df.columns)} total columns")
        if len(sc_columns) > 0:
            self.logger.info(f"Found {len(sc_columns)} scaffold columns")
        else:
            self.logger.warning(f"No scaffold columns found for type: {substructure_type}")
        if len(sub_columns) > 0:
            self.logger.info(f"Found {len(sub_columns)} substituent columns")
        
        # Reindex the dataframe
        final_df = final_df.reindex(columns=final_columns)
        
        # Make sure we actually have data to save
        if len(final_df) == 0:
            self.logger.error("No data to save! DataFrame is empty.")
            # Save an empty file with just column headers as a fallback
            final_df = pd.DataFrame(columns=final_columns)
            final_df.to_csv(f"{output_csv}.empty", index=False)
            raise ValueError("No data to save - check your processing steps")
        
        # Save the results
        try:
            if output_csv.lower().endswith('.parquet'):
                # Check for problematic columns before saving
                for col in final_df.columns:
                    if final_df[col].dtype == 'object':
                        sample = final_df[col].dropna().iloc[0] if not final_df[col].dropna().empty else None
                        if isinstance(sample, (list, np.ndarray)):
                            self.logger.warning(f"Column {col} contains complex objects - converting to string")
                            final_df[col] = final_df[col].apply(lambda x: str(x) if x is not None else None)
                
                # Now save the parquet
                final_df.to_parquet(output_csv, index=False)
            else:
                final_df.to_csv(output_csv, index=False)
            
            self.logger.info(f"Results saved to {output_csv}")
            # Optional rules mining at end of process
            try:
                if getattr(self.config, 'MINE_RULES', False):
                    self._mine_rules(all_results, substructure_type, os.path.dirname(output_csv))
            except Exception as e:
                self.logger.warning(f"Rules mining failed: {e}")
            
            # Verify file size
            if os.path.exists(output_csv):
                file_size = os.path.getsize(output_csv)
                self.logger.info(f"Output file size: {file_size} bytes")
                if file_size == 0:
                    self.logger.error("Output file has zero size! Falling back to CSV.")
                    final_df.to_csv(output_csv + ".fallback.csv", index=False)
        except Exception as e:
            self.logger.error(f"Error saving to {output_csv}: {str(e)}")
            
            # Fallback to CSV if Parquet fails
            if output_csv.lower().endswith('.parquet'):
                csv_output = output_csv.replace('.parquet', '.csv')
                try:
                    final_df.to_csv(csv_output, index=False)
                    self.logger.info(f"Fallback: Results saved to {csv_output}")
                except Exception as csv_e:
                    self.logger.error(f"Failed to save even as CSV: {str(csv_e)}")
                    # Last resort - save the error partial
                    final_df.to_csv(f"{output_csv}.error_partial", index=False)

        # Cleanup temp
        tmp = f"{output_csv}.temp"
        if os.path.exists(tmp):
            os.remove(tmp)
            self.logger.info(f"Removed temporary file {tmp}")

    def _load_ffv_calibrator(self, calib_dir: str, enabled: bool = True):
        if not enabled:
            return None
        try:
            p = Path(getattr(self.config, 'output_dir', '.')) / str(calib_dir).strip('/')
            pkl = p / 'ffv_isotonic.pkl'
            npz = p / 'ffv_isotonic_fallback.npz'
            if joblib is not None and pkl.exists():
                return joblib.load(pkl)
            if npz.exists():
                data = np.load(npz)
                X, Y = data['X'], data['Y']
                def eval_iso(x):
                    x = np.asarray(x)
                    return np.interp(x, X, Y, left=Y[0], right=Y[-1])
                class _IsoFallback:
                    def predict(self, x):
                        return eval_iso(x)
                return _IsoFallback()
        except Exception as e:
            self.logger.warning(f'Failed to load FFV calibrator: {e}')
        return None

    def _mine_rules(self, all_results: List[Dict[str, Any]], substructure_type: Optional[str], out_dir: str) -> None:
        """Aggregate Murcko fragment attributions into simple FFV design rules.
        Writes CSV and a quick PNG bar plot of top +/- fragments by mean attribution.
        """
        try:
            if not substructure_type or substructure_type.lower() != 'murcko':
                self.logger.info('Rules mining currently supports murcko only; skipping')
                return
            rows = []
            for r in all_results:
                comp_id = r.get('COMPOUND_ID')
                # scaffolds
                i = 0
                while True:
                    attr_key = f'{substructure_type}_substructure_{i}_attribution'
                    smi_key = f'{substructure_type}_substructure_{i}_smiles'
                    if attr_key in r and smi_key in r:
                        try:
                            attr = float(r[attr_key])
                            smi = str(r[smi_key])
                            if smi:
                                rows.append({'unit_smiles': smi, 'unit_type': 'scaffold', 'attribution': attr, 'compound_id': comp_id})
                        except Exception:
                            pass
                        i += 1
                        continue
                    break
                # substituents
                i = 0
                while True:
                    any_found = False
                    j = 0
                    while True:
                        smi_key = f'{substructure_type}_substituent_{i}_{j}_smiles'
                        att_key = f'{substructure_type}_substituent_{i}_{j}_attribution'
                        if smi_key in r and att_key in r:
                            any_found = True
                            try:
                                smi = str(r[smi_key])
                                attr = float(r[att_key])
                                if smi:
                                    rows.append({'unit_smiles': smi, 'unit_type': 'substituent', 'attribution': attr, 'compound_id': comp_id})
                            except Exception:
                                pass
                            j += 1
                            continue
                        break
                    if not any_found:
                        break
                    i += 1
            if not rows:
                self.logger.warning('No fragment entries found for rules mining')
                return
            df = pd.DataFrame(rows)
            grp = df.groupby(['unit_smiles', 'unit_type'])['attribution']
            agg = grp.agg(['count', 'mean', 'median'])
            pos_frac = (grp.apply(lambda s: (s > 0).mean())).rename('frac_pos')
            res = agg.join(pos_frac)
            res['consistency'] = res['frac_pos'].apply(lambda x: max(x, 1.0 - x))
            res = res.rename(columns={'count': 'n_present', 'mean': 'mean_attr', 'median': 'median_attr'})
            res = res.reset_index()
            csv_path = os.path.join(out_dir, 'ffv_rules_from_screen.csv')
            res.to_csv(csv_path, index=False)
            self.logger.info(f'Wrote rules CSV: {csv_path}')
            # Plot top +/-
            try:
                import matplotlib.pyplot as plt
                # Top positives and negatives by mean_attr
                top_pos = res.sort_values('mean_attr', ascending=False).head(10)
                top_neg = res.sort_values('mean_attr', ascending=True).head(10)
                fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
                def _short(x: str, k: int = 18) -> str:
                    return x if len(x) <= k else x[:k-3] + '...'
                axes[0].barh([_short(s) for s in top_pos['unit_smiles']], top_pos['mean_attr'], color='#2ca02c')
                axes[0].set_title('Top + mean attribution')
                axes[0].invert_yaxis()
                axes[1].barh([_short(s) for s in top_neg['unit_smiles']], top_neg['mean_attr'], color='#d62728')
                axes[1].set_title('Top - mean attribution')
                axes[1].invert_yaxis()
                plt.tight_layout()
                png_path = os.path.join(out_dir, 'ffv_rules_from_screen.png')
                plt.savefig(png_path, dpi=200)
                plt.close()
                self.logger.info(f'Wrote rules PNG: {png_path}')
            except Exception as pe:
                self.logger.warning(f'Rules PNG generation failed: {pe}')
        except Exception as e:
            self.logger.warning(f'Rules mining exception: {e}')

    def _clean_memory(self):
        """Explicitly clean up memory to prevent OOM errors."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()            
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f} minutes"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {minutes:.0f}m"




def main():
    # Initialize config and logger
    config = Configuration()
    logger = get_logger(__name__)

    # Argument parsing
    parser = argparse.ArgumentParser(description='Run polymer property predictions with XAI.')
    default_input = os.path.join(config.data_dir, 'polymer_candidates.csv')
    # Accept both --input_csv/--output_csv and aliases --input/--output
    parser.add_argument('--input_csv', type=str, default=default_input, help='Path to input file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to save prediction results')
    parser.add_argument('--input', dest='input_csv', type=str, help='Alias for --input_csv')
    parser.add_argument('--output', dest='output_csv', type=str, help='Alias for --output_csv')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None, help='Device to use')
    parser.add_argument('--substructure_type', type=str, choices=['murcko', 'None'], default='murcko')
    # Allow both flag and explicit 0/1 value for compatibility with wrappers
    parser.add_argument('--emit_xai_json', nargs='?', const=1, type=int, choices=[0,1], default=0, help='Emit XAI JSON (1 to enable)')
    parser.add_argument('--ensemble', type=str, choices=['best5','full'], default='best5')
    # Optional representation control flags (non-breaking)
    parser.add_argument('--predict_repr', type=str, choices=['auto','smiles','bigsmiles'], default=None, help='Representation to use for prediction inputs (default: config.PREDICT_REPR)')
    parser.add_argument('--bigsmiles_required', type=int, choices=[0,1], default=None, help='If 1, skip molecules that fail BigSMILES conversion')
    parser.add_argument('--dry_run', type=int, default=0, help='If >0, sample N molecules and report BigSMILES conversion stats, then exit')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None, help='End index (inclusive)')
    parser.add_argument('--resume', action='store_true', help='Resume from last saved position')
    parser.add_argument('--mine_rules', type=int, default=0, help='Enable rules mining (1 to enable)')
    
    args = parser.parse_args()

    # Handle None substructure type
    if args.substructure_type == 'None':
        args.substructure_type = None

    # Determine device with robust check
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
    if args.device == 'cuda':
        try:
            if torch.cuda.device_count() == 0:
                raise RuntimeError('No CUDA devices available')
        except Exception as e:
            logger.warning(f'CUDA not usable ({e}); falling back to CPU')
            args.device = 'cpu'

    logger.info('='*60)
    logger.info(f'Starting Polymer Property Prediction Process on device: {args.device}')
    logger.info('='*60)

    # Prepare output path
    if args.output_csv is None:
        base = os.path.splitext(os.path.basename(args.input_csv))[0]
        ext = '.csv' if args.input_csv.lower().endswith('.csv') else '.parquet'
        subtype_suffix = f"_{args.substructure_type}" if args.substructure_type else "_no_xai"
        args.output_csv = os.path.join(os.path.dirname(args.input_csv), 
                                      f'{base}_prediction_regression{subtype_suffix}{ext}')
    
    out_dir = os.path.dirname(args.output_csv)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f'Output will be written to {args.output_csv}')

    # Representation flags from CLI take precedence if provided
    if args.predict_repr is not None:
        config.PREDICT_REPR = args.predict_repr
    if args.bigsmiles_required is not None:
        config.BIGSMILES_REQUIRED = bool(int(args.bigsmiles_required))

    # Optional dry run for conversion stats
    if int(getattr(args, 'dry_run', 0)) > 0:
        try:
            n = int(args.dry_run)
            df_preview = pd.read_parquet(args.input_csv) if args.input_csv.lower().endswith('.parquet') else pd.read_csv(args.input_csv)
            df_preview = df_preview.dropna(subset=['SMILES']).head(n)
            mode = str(getattr(config, 'PREDICT_REPR', 'auto') or 'auto').lower()
            require_bigs = bool(getattr(config, 'BIGSMILES_REQUIRED', False))
            n_ok = 0
            n_total = 0
            for _, row in df_preview.iterrows():
                s = str(row['SMILES'])
                try:
                    _, _, ok = roundtrip_or_fallback(s) if mode in ('auto','bigsmiles') else (s, s, False)
                except Exception:
                    ok = False
                n_total += 1
                if ok:
                    n_ok += 1
            pct = (n_ok / n_total * 100.0) if n_total else 0.0
            logger.info(f"Dry run BigSMILES coverage: {n_ok}/{n_total} ({pct:.1f}%) | mode={mode} required={int(require_bigs)}")
            return
        except Exception as e:
            logger.warning(f"Dry run failed: {e}")

    # Respect CLI flag for XAI JSON
    if int(getattr(args, 'emit_xai_json', 0)) == 1:
        config.EMIT_XAI_JSON = True
    # Respect CLI rules mining
    if int(getattr(args, 'mine_rules', 0)) == 1:
        config.MINE_RULES = True

    # Initialize PredictionManager with CPU fallback on model-load failure
    try:
        mgr = PredictionManager(
            config=config, 
            input_csv=args.input_csv, 
            device=args.device, 
            num_workers=args.num_workers, 
            ensemble_type=args.ensemble
        )
    except Exception as e:
        # If CUDA model load fails, retry on CPU
        logger.warning(f'Model init failed on {args.device}: {e}. Retrying on CPU')
        args.device = 'cpu'
        mgr = PredictionManager(
            config=config, 
            input_csv=args.input_csv, 
            device='cpu', 
            num_workers=args.num_workers, 
            ensemble_type=args.ensemble
        )

    try:
        mgr.process_dataset(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            batch_size=args.batch_size,
            substructure_type=args.substructure_type,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            resume=args.resume
        )
        
        # Create done marker
        if args.start_idx != 0 or args.end_idx is not None:
            shard = f"{args.start_idx}_{args.end_idx}"
        else:
            shard = "complete"
        done_file = os.path.join(out_dir, f'prediction_{shard}.done')
        with open(done_file, 'w') as f:
            f.write(datetime.now().isoformat())
        logger.info(f'Created done marker: {done_file}')
        logger.info('Polymer prediction completed successfully')
        
    except Exception as e:
        logger.error(f'Prediction failed: {e}')
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()



