# xai_viz.py
import math
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

# Using existing PredictionManager implementation
# from predict.py, which exposes:
#  - get_substructures(smiles, 'murcko' | 'brics' | 'fg') -> {id: [atom_idx,...]}
#  - _compute_mask_deltas(smiles, mask_list, target_name) -> [{'atoms': [...], 'raw_delta': .., 'norm_delta': .., 'size': ..}, ...]
#  - _choose_text_repr(smiles, mode='auto', require_bigs=False) to BigSMILES-roundtrip if available
#  - predict_molecule_with_confidence/similar if you want FFV_mean/std

FFV_TARGET = 'FFV'

# --- Optional isotonic calibration loader ---
def _maybe_load_isotonic(cal_path: str | None):
    if not cal_path:
        return None
    try:
        from joblib import load
        import os
        if os.path.exists(cal_path):
            return load(cal_path)
    except Exception:
        pass
    return None

def collect_masks(pm, smiles: str, substructure_type: str = 'murcko', topk: int = 12):
    """
    Build candidate fragment masks from the configured substructure type and
    prune to non-overlapping top-K using pm._select_non_overlapping_masks.
    """
    subs = pm.get_substructures(smiles, substructure_type) or {}
    masks = [{'mask_id': f'{substructure_type.upper()}_{i:02d}',
              'atoms': list(map(int, atoms)),
              'mask_family': 'fragment',
              'source': substructure_type} for i, atoms in subs.items()]
    # select disjoint masks (reuse the manager's greedy)
    selected = pm._select_non_overlapping_masks(masks, topk)
    return selected

def rank_masks_by_delta(pm, smiles: str, masks: List[Dict], normalize: bool = True) -> List[Dict]:
    """
    Compute Δ(FFV) per mask via ensemble mean differences using your existing
    masked-pred routine. Positive delta => mask contributes to higher FFV.
    """
    augmented = pm._compute_mask_deltas(smiles, masks, FFV_TARGET) or []
    if not augmented:
        return []

    # choose which delta to rank/show
    key = 'norm_delta' if normalize else 'raw_delta'
    ranked = sorted(augmented, key=lambda m: abs(m.get(key, 0.0)), reverse=True)
    # attach 'delta' field to unify downstream
    for r in ranked:
        r['delta'] = float(r.get(key, 0.0))
    return ranked

def split_pos_neg(ranked: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    pos = [m for m in ranked if m['delta'] > 0]
    neg = [m for m in ranked if m['delta'] < 0]
    return pos, neg

def _to_color(val: float) -> Tuple[float, float, float]:
    """
    Map sign to color: blue for positive (↑FFV), red for negative (↓FFV).
    The magnitude scales the alpha used for highlighting.
    """
    if val >= 0:
        # blue
        return (0.10, 0.35, 0.95)
    # red
    return (0.90, 0.20, 0.15)

def draw_ffv_attributions(smiles: str,
                          ranked_masks: List[Dict],
                          max_atoms: int = 200,
                          legend: Optional[str] = None,
                          size: Tuple[int, int] = (900, 600),
                          legend_scale: float = 1.6,
                          show_atom_indices: bool = True,
                          palette: str = "blue_orange"):
    """
    Render the molecule and color atoms by signed attribution.
    Blue = ↑FFV (positive delta), Red = ↓FFV (negative delta).
    Falls back to PIL overlay if RDKit lacks DrawLegend().
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    import numpy as np
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES/BigSMILES (after fallback).")

    if mol.GetNumAtoms() > max_atoms:
        raise ValueError(f"Molecule too large to draw nicely (>{max_atoms} atoms).")

    # aggregate atom contributions
    per_atom = np.zeros(mol.GetNumAtoms(), dtype=float)
    for m in ranked_masks:
        delta = float(m.get('delta', 0.0))
        for a in m.get('atoms', []):
            if 0 <= a < mol.GetNumAtoms():
                per_atom[a] += delta

    # highlight setup - use palette system
    pal = PALETTES.get(palette, PALETTES["blue_orange"])
    highlight_atoms = [i for i, v in enumerate(per_atom) if abs(v) > 1e-9]
    atom_cols, atom_rads = {}, {}
    if highlight_atoms:
        scale_den = np.percentile(np.abs(per_atom[highlight_atoms]), 95)
        scale_den = scale_den if scale_den > 0 else 1.0
        for idx in highlight_atoms:
            v = per_atom[idx]
            col = pal["pos"] if v >= 0 else pal["neg"]  # use palette colors
            atom_cols[idx] = col
            atom_rads[idx] = 0.4 + 0.6 * min(1.0, abs(v) / scale_den)

    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.fixedBondLength = 25
    
    # Add atom indices if requested:
    if show_atom_indices:
        try:
            # RDKit 2020.09+ : this draws small index numbers near atoms
            opts.addAtomIndices = True
            # optional: make them a touch larger
            opts.annotationFontScale = 1.2
        except Exception:
            # Fallback for older RDKit: we manually annotate each atom
            # (we'll place the numbers after drawing the molecule)
            pass
    else:
        opts.addAtomIndices = False

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_cols,
        highlightAtomRadii=atom_rads
    )

    # Manual overlay for older RDKit if addAtomIndices didn't exist:
    if show_atom_indices and not getattr(opts, "addAtomIndices", False):
        try:
            for a in mol.GetAtoms():
                p = drawer.GetDrawCoords(a.GetIdx())
                # nudge the text a bit so it doesn't clash with the atom symbol
                drawer.DrawString(str(a.GetIdx()), p[0] + 2, p[1] + 2)
        except Exception:
            # If GetDrawCoords or DrawString don't exist, skip manual overlay
            pass

    drawer.FinishDrawing()
    png = drawer.GetDrawingText()

    # Convert to PIL and add legend overlay (since RDKit legend methods are inconsistent)
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(BytesIO(png)).convert("RGBA")
    
    if legend:
        draw = ImageDraw.Draw(im, "RGBA")
        
        # Scale font size based on legend_scale
        base_font_size = 14
        font_size = int(base_font_size * legend_scale)
        
        try:
            # Try to load a better font
            font = ImageFont.load_default()
            if hasattr(ImageFont, 'truetype'):
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    pass
        except Exception:
            font = None
        
        # Calculate text dimensions
        if font:
            try:
                bbox = draw.textbbox((0, 0), legend, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except:
                tw, th = len(legend) * font_size * 0.6, font_size
        else:
            tw, th = len(legend) * font_size * 0.6, font_size
        
        # Create legend box with padding
        pad = int(font_size * 0.4)
        box_w = int(tw + 2 * pad)
        box_h = int(th + 2 * pad)
        
        # Position at top-left with margin
        margin = 8
        x0, y0 = margin, margin
        x1, y1 = x0 + box_w, y0 + box_h
        
        # Draw semi-transparent background with border
        draw.rectangle([x0, y0, x1, y1], 
                      fill=(255, 255, 255, 220), 
                      outline=(0, 0, 0, 255), 
                      width=1)
        
        # Draw text
        text_x = x0 + pad
        text_y = y0 + pad
        draw.text((text_x, text_y), legend, 
                 fill=(0, 0, 0, 255), 
                 font=font)

    return im

# Palettes (RGB in 0..1 floats for RDKit)
PALETTES: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "blue_orange": {
        "pos": (0x2B/255, 0x8C/255, 0xBE/255),   # +Δ (↑FFV)
        "neg": (0xE6/255, 0x61/255, 0x01/255),   # −Δ (↓FFV)
    },
    "red_blue": {
        "pos": (0x2B/255, 0x8C/255, 0xBE/255),
        "neg": (0xD1/255, 0x39/255, 0x5F/255),
    },
}

def _per_atom_scores_from_masks(n_atoms: int, ranked_masks: List[dict]) -> List[float]:
    """Aggregate signed mask deltas to a quick per-atom score for tooltips/labels."""
    scores = [0.0]*n_atoms
    for m in ranked_masks:
        w = float(m.get("delta", 0.0))
        for a in m.get("atoms", []):
            scores[a] += w
    return scores

def draw_ffv_attributions_svg(
    smiles: str,
    ranked_masks: List[dict],
    legend: str = "",
    size: Tuple[int, int] = (900, 600),
    palette: str = "blue_orange",
    show_atom_indices: bool = True,
    legend_scale: float = 1.2,
) -> str:
    """
    Like draw_ffv_attributions(...), but returns an SVG string with:
      - color-blind friendly palette (default: blue_orange)
      - optional atom indices
      - per-atom hover tooltips (id + aggregated Δ)
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
    import re

    pal = PALETTES.get(palette, PALETTES["blue_orange"])

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Cannot parse SMILES for SVG drawing.")
    rdDepictor.Compute2DCoords(mol)
    n_atoms = mol.GetNumAtoms()

    # build per-atom colors/radii by taking the *largest magnitude* mask for each atom
    atom_colors = {}
    atom_radii = {}
    chosen = {}  # atom -> (absDelta, sign)

    for m in ranked_masks:
        d = float(m.get("delta", 0.0))
        if d == 0.0: 
            continue
        col = pal["pos"] if d > 0 else pal["neg"]
        rad = 0.35  # visual weight bubble size
        for a in m.get("atoms", []):
            prev = chosen.get(a, (0.0, 0))
            if abs(d) >= prev[0]:
                chosen[a] = (abs(d), 1 if d > 0 else -1)
                atom_colors[a] = col
                atom_radii[a] = rad

    drawer = MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    # larger legend text
    opts.legendFontSize = int(legend_scale * opts.legendFontSize)
    # atom indices if requested
    if show_atom_indices:
        opts.addAtomIndices = True
        opts.annotationFontScale = 1.6

    highlight_atoms = list(atom_colors.keys())
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    # Skip DrawAnnotation as it's not available in all RDKit versions
    # Legend will be added via PIL overlay instead
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Inject hover tooltips: <title> inside each atom group
    # First compute per-atom aggregated scores:
    per_atom = _per_atom_scores_from_masks(n_atoms, ranked_masks)

    def inject_tooltip(svg_text: str) -> str:
        # RDKit tags atoms like class='atom-#' or 'atom-#' in group ids.
        for idx in range(n_atoms):
            # human-friendly signed value
            val = per_atom[idx]
            sign = "+" if val > 0 else ""
            tooltip = f"Atom {idx} | Δ={sign}{val:.4f}"
            # Regex: find opening tag of atom group and inject <title>...</title>
            # Covers both id="atom-#" and class="atom-#"
            pattern = rf"(<g[^>]*?(?:id|class)=[\"']atom-{idx}[\"'][^>]*>)"
            repl = rf"\1<title>{tooltip}</title>"
            svg_text = re.sub(pattern, repl, svg_text, count=1)
        return svg_text

    svg = inject_tooltip(svg)
    return svg

def explain_and_draw_svg(pm, smiles_or_bigsmiles, substructure_type="murcko",
                         topk=12, normalize=True, palette="blue_orange",
                         show_atom_indices=True, use_calibration=True,
                         calibration_path="output/calibration/ffv_isotonic.pkl",
                         legend_scale=1.6, return_data=False):
    """Thin wrapper to generate SVG with tooltips using existing explain_and_draw logic."""
    img, meta = explain_and_draw(pm, smiles_or_bigsmiles, substructure_type, topk, normalize, 
                               use_calibration=use_calibration, calibration_path=calibration_path,
                               legend_scale=legend_scale, show_atom_indices=show_atom_indices,
                               return_data=True)
    # reuse canon SMILES + ranked masks from meta to build SVG with tooltips
    canon = meta["canon_smiles"]
    ranked = meta["topk_fragments"]
    # legend from previous call if you build it there; otherwise:
    legend = meta.get("legend", f"FFV {meta.get('ffv', '—')}")
    svg = draw_ffv_attributions_svg(
        canon, ranked, legend=legend,
        palette=palette, show_atom_indices=show_atom_indices
    )
    if return_data:
        return svg, meta
    return svg

def explain_and_draw(pm,
                     smiles_or_bigsmiles: str,
                     substructure_type: str = 'murcko',
                     topk: int = 12,
                     normalize: bool = True,
                     use_calibration: bool = True,
                     calibration_path: str | None = "output/calibration/ffv_isotonic.pkl",
                     legend_scale: float = 1.6,
                     show_atom_indices: bool = True,
                     palette: str = "blue_orange",
                     return_data: bool = False):
    """
    End-to-end:
      - choose BigSMILES if available (auto),
      - build disjoint masks,
      - compute Δ(FFV),
      - draw colored molecule (blue=↑FFV, red=↓FFV).
    """
    # Canonicalize & text repr (auto → BigSMILES if roundtrip OK)
    canon, text_repr, used_bigs = pm._choose_text_repr(smiles_or_bigsmiles, mode='auto', require_bigs=False)

    # collect masks & deltas
    masks = collect_masks(pm, canon, substructure_type=substructure_type, topk=topk)
    ranked = rank_masks_by_delta(pm, canon, masks, normalize=normalize)

    # make a compact legend
    ffv_pred = None
    try:
        pred = pm.predict_molecule_with_confidence(canon) or {}
        ffv_pred = pred.get('pred_FFV') or pred.get('ffv_pred')
    except Exception:
        pass
    
    cal = _maybe_load_isotonic(calibration_path) if use_calibration else None
    if ffv_pred is not None:
        if cal is not None:
            try:
                ffv_cal = float(cal.predict([[ffv_pred]])[0])
                legend = f"FFV {ffv_cal:.3f} (raw {ffv_pred:.3f}) | repr={'BigSMILES' if used_bigs else 'SMILES'}"
            except Exception:
                legend = f"FFV {ffv_pred:.3f} | repr={'BigSMILES' if used_bigs else 'SMILES'}"
        else:
            legend = f"FFV {ffv_pred:.3f} | repr={'BigSMILES' if used_bigs else 'SMILES'}"
    else:
        legend = f"repr={'BigSMILES' if used_bigs else 'SMILES'}"

    img = draw_ffv_attributions(canon, ranked, legend=legend, legend_scale=legend_scale, show_atom_indices=show_atom_indices, palette=palette)

    if return_data:
        return img, {
            'repr_used': 'BigSMILES' if used_bigs else 'SMILES',
            'text_repr': text_repr,
            'canon_smiles': canon,
            'topk_fragments': [{
                'mask_id': m.get('mask_id', ''),
                'atoms': m.get('atoms', []),
                'delta': float(m['delta']),
                'size': int(m.get('size', 0))
            } for m in ranked]
        }
    return img
