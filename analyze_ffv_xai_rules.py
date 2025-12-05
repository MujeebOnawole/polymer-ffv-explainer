import os
import re
import argparse
import json
from typing import List, Dict

import numpy as np
import pandas as pd


def find_fragment_columns(df: pd.DataFrame, prefix: str) -> List[int]:
    idxs = set()
    pat = re.compile(rf"{re.escape(prefix)}_(\d+)_")
    for c in df.columns:
        m = pat.search(c)
        if m:
            idxs.add(int(m.group(1)))
    return sorted(list(idxs))


def melt_fragments(df: pd.DataFrame, prefix: str = 'murcko_substructure') -> pd.DataFrame:
    idxs = find_fragment_columns(df, prefix)
    rows = []
    for i in idxs:
        attr_col = f"{prefix}_{i}_FFV_attribution"
        smi_col = f"{prefix}_{i}_smiles"
        if attr_col not in df.columns or smi_col not in df.columns:
            continue
        sub = df[[
            'COMPOUND_ID', 'SMILES', 'ffv_pred', 'FFV_std', 'FFV_confidence', 'xai_status',
            attr_col, smi_col
        ]].copy()
        sub = sub.rename(columns={attr_col: 'fragment_attr', smi_col: 'fragment_smiles'})
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=['COMPOUND_ID','SMILES','ffv_pred','FFV_std','FFV_confidence','xai_status','fragment_attr','fragment_smiles'])
    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out.dropna(subset=['fragment_smiles', 'fragment_attr'])
    # normalize fragment SMILES strings (strip empty strings)
    out['fragment_smiles'] = out['fragment_smiles'].astype(str).str.strip()
    out = out[out['fragment_smiles'] != '']
    return out


def main():
    parser = argparse.ArgumentParser(description='Mine FFV XAI rules from predictions + training truth.')
    parser.add_argument('--pred_csv', type=str, required=True, help='Path to external_with_preds_4edge.csv')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to polymer_properties.csv (with true FFV)')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory (default: FFV_XAI next to pred_csv)')
    parser.add_argument('--min_support', type=int, default=5, help='Minimum molecules per fragment to report')
    parser.add_argument('--confidence', type=str, default='high', choices=['high','medium','low','any'], help='Model-side XAI confidence to keep (if present)')
    parser.add_argument('--level_filter', type=str, default='none', choices=['A','B','C','D','none'], help='Filter by ordinal error level using true FFV (A=<=5%, B=<=10%, C=<=20% of range)')
    parser.add_argument('--ffv_min', type=float, default=None, help='FFV minimum for ordinal thresholds (default: infer from train or use 0.23)')
    parser.add_argument('--ffv_max', type=float, default=None, help='FFV maximum for ordinal thresholds (default: infer from train or use 0.78)')
    parser.add_argument('--prefix', type=str, default='murcko_substructure', help='Fragment prefix to search (default murcko_substructure)')
    args = parser.parse_args()

    pred_path = args.pred_csv
    train_path = args.train_csv
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(pred_path), 'FFV_XAI')
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    df_pred = pd.read_csv(pred_path)
    df_train = pd.read_csv(train_path)

    # Prepare truth mapping
    # Ensure id types align
    df_pred['COMPOUND_ID'] = df_pred['COMPOUND_ID'].astype(str)
    df_train['id'] = df_train['id'].astype(str)
    truth = df_train[['id', 'FFV']].rename(columns={'id':'COMPOUND_ID', 'FFV':'FFV_true'})

    # Merge truth
    df_pred = df_pred.merge(truth, on='COMPOUND_ID', how='left')

    # Melt fragments
    fr = melt_fragments(df_pred, prefix=args.prefix)
    if fr.empty:
        raise SystemExit('No fragment columns found in predictions CSV. Expected columns like murcko_substructure_0_smiles and murcko_substructure_0_FFV_attribution')

    # Attach truth and residuals
    fr = fr.merge(df_pred[['COMPOUND_ID','FFV_true']], on='COMPOUND_ID', how='left')
    fr['residual'] = fr['ffv_pred'] - fr['FFV_true']

    # Compute ordinal levels (A/B/C/D) based on absolute error vs FFV range
    try:
        ffv_min = float(args.ffv_min) if args.ffv_min is not None else float(np.nanmin(df_train['FFV'].values))
    except Exception:
        ffv_min = 0.23
    try:
        ffv_max = float(args.ffv_max) if args.ffv_max is not None else float(np.nanmax(df_train['FFV'].values))
    except Exception:
        ffv_max = 0.78
    delta = max(1e-12, ffv_max - ffv_min)
    thrA, thrB, thrC = 0.05 * delta, 0.10 * delta, 0.20 * delta
    fr['abs_err'] = np.abs(fr['ffv_pred'] - fr['FFV_true'])
    def _level(e):
        if pd.isna(e):
            return None
        if e <= thrA:
            return 'A'
        if e <= thrB:
            return 'B'
        if e <= thrC:
            return 'C'
        return 'D'
    fr['ordinal_level'] = fr['abs_err'].apply(_level)

    # Reliability filter
    mask = (fr['xai_status'].astype(str).str.lower() == 'completed')
    if args.confidence != 'any' and 'FFV_confidence' in fr.columns:
        mask &= (fr['FFV_confidence'].astype(str).str.lower() == args.confidence)
    if args.level_filter != 'none':
        mask &= (fr['ordinal_level'] == args.level_filter)
    fr_rel = fr[mask].copy()

    # Save instance-level tables for auditing
    cols_keep = ['fragment_smiles','COMPOUND_ID','SMILES','FFV_true','ffv_pred','residual','fragment_attr','FFV_confidence','ordinal_level','abs_err']
    inst_all = fr_rel[cols_keep].copy()
    inst_all.to_csv(os.path.join(args.out_dir, 'ffv_fragment_instances.csv'), index=False)
    pos = inst_all[inst_all['fragment_attr'] > 0].copy()
    neg = inst_all[inst_all['fragment_attr'] < 0].copy()
    pos.to_csv(os.path.join(args.out_dir, 'ffv_fragment_positive_instances.csv'), index=False)
    neg.to_csv(os.path.join(args.out_dir, 'ffv_fragment_negative_instances.csv'), index=False)

    # Aggregate rules by fragment_smiles
    def _safe_mean(x):
        try:
            return float(np.nanmean(x))
        except Exception:
            return float('nan')

    rules = (
        fr_rel.groupby('fragment_smiles')
        .agg(
            n_molecules=('COMPOUND_ID', 'nunique'),
            n_instances=('COMPOUND_ID', 'size'),
            mean_attr=('fragment_attr', 'mean'),
            median_attr=('fragment_attr', 'median'),
            std_attr=('fragment_attr', lambda s: s.std(ddof=1)),
            pos_frac=('fragment_attr', lambda s: (s > 0).mean()),
            mean_residual=('residual', _safe_mean),
            mean_pred_ffv=('ffv_pred', _safe_mean),
        )
        .reset_index()
    )

    # Support filter
    rules = rules[rules['n_molecules'] >= int(args.min_support)].copy()

    # Direction labels
    rules['direction'] = np.where(rules['mean_attr'] > 0, 'increase', np.where(rules['mean_attr'] < 0, 'decrease', 'neutral'))

    # Sort and save summaries
    rules_all_path = os.path.join(args.out_dir, 'ffv_fragment_rules_summary.csv')
    rules.sort_values('mean_attr', ascending=False).to_csv(rules_all_path, index=False)

    inc = rules[rules['mean_attr'] > 0].sort_values('mean_attr', ascending=False)
    dec = rules[rules['mean_attr'] < 0].sort_values('mean_attr')
    inc.to_csv(os.path.join(args.out_dir, 'ffv_fragment_rules_top_increasing.csv'), index=False)
    dec.to_csv(os.path.join(args.out_dir, 'ffv_fragment_rules_top_decreasing.csv'), index=False)

    # Compact JSON summary for quick inspection
    top_k = 20
    # Build example instances for top rules
    def top_examples(df: pd.DataFrame, frag: str, sign: str, k: int = 10) -> List[Dict]:
        sub = df[df['fragment_smiles'] == frag].copy()
        if sign == 'increase':
            sub = sub.sort_values('fragment_attr', ascending=False)
        else:
            sub = sub.sort_values('fragment_attr', ascending=True)
        sub = sub.head(k)
        return [
            {
                'COMPOUND_ID': r['COMPOUND_ID'],
                'SMILES': r['SMILES'],
                'FFV_true': None if pd.isna(r['FFV_true']) else float(r['FFV_true']),
                'FFV_pred': None if pd.isna(r['ffv_pred']) else float(r['ffv_pred']),
                'residual': None if pd.isna(r['residual']) else float(r['residual']),
                'attr': None if pd.isna(r['fragment_attr']) else float(r['fragment_attr'])
            }
            for _, r in sub.iterrows()
        ]

    summary = {
        'source_pred_csv': os.path.abspath(pred_path),
        'source_train_csv': os.path.abspath(train_path),
        'min_support': int(args.min_support),
        'confidence': args.confidence,
        'n_instances': int(len(fr_rel)),
        'n_fragments': int(rules.shape[0]),
        'top_increasing': [
            {
                **row,
                'examples': top_examples(inst_all, row['fragment_smiles'], 'increase', 10)
            }
            for row in inc.head(top_k).to_dict(orient='records')
        ],
        'top_decreasing': [
            {
                **row,
                'examples': top_examples(inst_all, row['fragment_smiles'], 'decrease', 10)
            }
            for row in dec.head(top_k).to_dict(orient='records')
        ],
    }
    with open(os.path.join(args.out_dir, 'ffv_fragment_rules_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"[FFV_XAI] Wrote: {rules_all_path}")


if __name__ == '__main__':
    main()
