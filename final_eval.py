import os
import io
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import pickle
from pathlib import Path
try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    IsotonicRegression = None
try:
    import joblib
except Exception:
    joblib = None


from model import BaseGNN
from data_module import MoleculeDataModule as MoleculeDataModule
from logger import LoggerSetup, get_logger
from config import Configuration


class FinalEvaluator:
    """Evaluate cross-validation models for polymer regression."""

    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.device = config.device
        self.property_names = config.property_names
        self.weights = config.competition_weights

        self.cv_dir = os.path.join(
            config.output_dir, f"{config.task_name}_{config.task_type}_cv_results"
        )
        self.eval_dir = os.path.join(
            config.output_dir, f"{config.task_name}_{config.task_type}_final_eval"
        )
        os.makedirs(self.eval_dir, exist_ok=True)
        LoggerSetup.initialize(self.eval_dir, f"{config.task_name}_final_eval")
        self._logger = get_logger(__name__)

        self.data_module = MoleculeDataModule(config)
        self.data_module.setup()
        self.test_dataset = self.data_module.test_dataset

        self.hyperparams = self._load_hyperparameters()
        self.best_models_df = self._load_best_model_info()

        self.individual_results: List[Dict] = []
        self.all_predictions: List[np.ndarray] = []
        self.labels: Optional[np.ndarray] = None
        self.ensemble_results: Optional[Dict] = None
        self.ensemble_preds_mean: Optional[np.ndarray] = None
        self.ensemble_preds_std: Optional[np.ndarray] = None
        self.selected_models: Optional[List[Dict]] = None
        # Per-property ensembles (ordinal Category-A driven)
        self.selected_models_by_property: Dict[str, List[Dict]] = {}
        self.ensemble_results_by_property: Dict[str, Dict] = {}
        self.ensemble_preds_mean_by_property: Dict[str, np.ndarray] = {}
        # Controls
        self.prop_ens_k: int = int(getattr(self.config, 'PROP_ENSEMBLE_K', 5))
        self.prop_ens_enforce_cv: bool = bool(getattr(self.config, 'PROP_ENSEMBLE_ENFORCE_CV', True))
        # Stacking paths
        self.stack_dir = os.path.join(self.config.output_dir, 'stack_models')
        self.stack_meta_dir = os.path.join(self.config.output_dir, 'stack_meta')
        self.stack_shap_dir = os.path.join(self.config.output_dir, self.config.STACK_SHAP_OUTDIR.strip('/'))
        os.makedirs(self.stack_dir, exist_ok=True)
        os.makedirs(self.stack_meta_dir, exist_ok=True)
        os.makedirs(self.stack_shap_dir, exist_ok=True)

    def _load_hyperparameters(self) -> Dict:
        path = os.path.join(
            self.cv_dir,
            f"{self.config.task_name}_{self.config.task_type}_final_results.json",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Final results JSON not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        return data["config"]["hyperparameters"]

    def _load_best_model_info(self) -> pd.DataFrame:
        csv_path = os.path.join(
            self.cv_dir,
            f"{self.config.task_name}_{self.config.task_type}_all_run_metrics.csv",
        )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        if "finalsaved" not in df.columns:
            raise ValueError("Metrics CSV missing 'FinalSaved' column")
        best = df[df["finalsaved"].str.lower() == "yes"].copy()
        if best.empty:
            raise ValueError("No models marked as FinalSaved in CSV")
        self._logger.info(f"Loaded {len(best)} best models from CSV")
        return best

    def _load_model_checkpoint(self, cv: int, fold: int) -> BaseGNN:
        ckpt_path = os.path.join(
            self.cv_dir,
            f"cv{cv}",
            "checkpoints",
            f"{self.config.task_name}_{self.config.task_type}_cv{cv}_fold{fold}_best.ckpt",
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        hparams = ckpt["hyperparameters"]
        model = BaseGNN(
            config=self.config,
            rgcn_hidden_feats=hparams["rgcn_hidden_feats"],
            ffn_hidden_feats=hparams["ffn_hidden_feats"],
            ffn_dropout=hparams["ffn_dropout"],
            rgcn_dropout=hparams["rgcn_dropout"],
            classification=False,
            num_classes=None,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _get_test_loader(self):
        return self.data_module.test_dataloader()

    def _get_predictions(self, model: BaseGNN, loader) -> tuple[np.ndarray, np.ndarray]:
        preds = {p: [] for p in self.property_names}
        labels_list = []
        with torch.no_grad():
            for graphs, labels in loader:
                graphs = graphs.to(self.device)
                outputs, _ = model(graphs)
                for prop in self.property_names:
                    preds[prop].extend(outputs[prop].cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        labels = np.vstack(labels_list)
        preds_arr = np.stack([preds[p] for p in self.property_names], axis=1)
        return preds_arr, labels

    def _calculate_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict:
        from sklearn.metrics import r2_score
        metrics: Dict[str, float] = {}
        total_loss = 0.0
        total_weight = 0.0
        r2_values = []
        valid_r2_count = 0

        for idx, prop in enumerate(self.property_names):
            mask = np.isfinite(labels[:, idx])
            n = int(mask.sum())
            if n > 0:
                mae = mean_absolute_error(labels[mask, idx], preds[mask, idx])
                if n >= 2:
                    r2 = r2_score(labels[mask, idx], preds[mask, idx])
                else:
                    r2 = float('nan')
                    try:
                        self._logger.info(f"R2 skipped for {prop}: insufficient samples (n={n})")
                    except Exception:
                        pass
                r2_values.append(r2)
                valid_r2_count += 1
            else:
                mae = 0.0
                r2 = float('nan')
            metrics[f"mae_{prop}"] = float(mae)
            metrics[f"r2_{prop}"] = float(r2)
            weight = self.weights[prop]
            total_loss += weight * mae
            total_weight += weight

        metrics["wmae"] = total_loss / total_weight if total_weight > 0 else 0.0
        # Mean RÂ² across properties with valid predictions
        metrics["r2"] = float(np.nanmean(r2_values)) if valid_r2_count > 0 else float('nan')
        return metrics

    def _evaluate_individual_models(self):
        for _, row in self.best_models_df.iterrows():
            cv = int(row["cv"])
            fold = int(row["fold"])
            loader = self._get_test_loader()
            model = self._load_model_checkpoint(cv, fold)
            preds, labels = self._get_predictions(model, loader)
            metrics = self._calculate_metrics(preds, labels)
            metrics.update({"cv": cv, "fold": fold})
            self.individual_results.append(metrics)
            self.all_predictions.append(preds)
            if self.labels is None:
                self.labels = labels
            self._logger.info(
                f"CV{cv} Fold{fold}: RÂ²={metrics['r2']:.4f}, wMAE={metrics['wmae']:.4f}"
            )
            del model
            torch.cuda.empty_cache()


    def _compute_ordinal_score_for_model(self, preds: np.ndarray) -> Dict:
        """Compute ordinal analysis scores for a single model's predictions.

        Returns dict with:
        - total_cat_a: total Category A predictions across all properties
        - property_cat_a: dict of Category A counts per property
        """
        scores = {
            'total_cat_a': 0,
            'property_cat_a': {},
        }

        for p_idx, prop in enumerate(self.property_names):
            y_true = self.labels[:, p_idx]
            y_pred = preds[:, p_idx]
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if not mask.any():
                scores['property_cat_a'][prop] = 0
                continue

            y_t = y_true[mask]
            y_p = y_pred[mask]

            # Compute threshold A (same logic as _run_ordinal_analysis_clean)
            rng = getattr(self.config, 'property_ranges', {}).get(prop) if hasattr(self.config, 'property_ranges') else None
            if rng is not None:
                delta = float(rng[1] - rng[0])
                thrA = 0.05 * delta
            else:
                abs_err_all = np.abs(y_p - y_t)
                thrA = np.quantile(abs_err_all, 0.25)

            # Count Category A predictions
            abs_err = np.abs(y_p - y_t)
            cat_a_count = int(np.sum(abs_err <= thrA))
            scores['property_cat_a'][prop] = cat_a_count
            scores['total_cat_a'] += cat_a_count

        return scores

    def _evaluate_ensemble(self):
        # CV-aware selection using ordinal analysis: ensure at least one model from each CV1, CV2, CV3
        if not self.individual_results or not self.all_predictions:
            raise ValueError('No individual model results available for ensembling')

        self._logger.info("Computing ordinal scores for all models to guide ensemble selection...")

        # Compute ordinal scores for all models
        ordinal_scores = []
        for i, preds in enumerate(self.all_predictions):
            score = self._compute_ordinal_score_for_model(preds)
            ordinal_scores.append(score)
            cv = self.individual_results[i].get('cv')
            fold = self.individual_results[i].get('fold')
            self._logger.info(f"  CV{cv} Fold{fold}: Total Cat A = {score['total_cat_a']}")

        # Build aligned entries
        entries = []
        for i, m in enumerate(self.individual_results):
            try:
                cv = int(m.get('cv'))
            except Exception:
                cv = m.get('cv')
            entries.append({
                'idx': i,
                'cv': cv,
                'fold': int(m.get('fold')) if 'fold' in m else None,
                'wmae': float(m.get('wmae', np.inf)),
                'preds': self.all_predictions[i],
                'total_cat_a': ordinal_scores[i]['total_cat_a'],
                'property_cat_a': ordinal_scores[i]['property_cat_a'],
            })

        # Group by CV
        from collections import defaultdict
        groups = defaultdict(list)
        for e in entries:
            groups[e['cv']].append(e)

        selected: List[dict] = []
        # Select best from each required CV based on ordinal score (total Category A)
        # Use wMAE as tiebreaker
        self._logger.info("Selecting best model from each CV based on Category A performance...")
        for required_cv in [1, 2, 3]:
            if required_cv in groups and groups[required_cv]:
                # Sort by total_cat_a (descending), then by wmae (ascending) as tiebreaker
                best = sorted(groups[required_cv], key=lambda x: (-x['total_cat_a'], x['wmae']))[0]
                selected.append(best)
                self._logger.info(f"CV{required_cv} best: Fold{best['fold']} (Cat A={best['total_cat_a']}, wMAE={best['wmae']:.4f})")
            else:
                self._logger.warning(f"No available models for CV{required_cv}; will fill from remaining best")

        # Fill remaining slots with next-best overall by ordinal score
        selected_indices = {e['idx'] for e in selected}
        remaining = [e for e in entries if e['idx'] not in selected_indices]
        remaining_sorted = sorted(remaining, key=lambda x: (-x['total_cat_a'], x['wmae']))
        self._logger.info("Filling remaining ensemble slots based on Category A performance...")
        for e in remaining_sorted:
            if len(selected) >= 5:
                break
            selected.append(e)
            self._logger.info(f"Added: CV{e['cv']} Fold{e['fold']} (Cat A={e['total_cat_a']}, wMAE={e['wmae']:.4f})")

        if len(selected) == 0:
            raise ValueError('Ensemble selection yielded zero models')
        if len(selected) < 5:
            self._logger.warning(f"Only {len(selected)} models available for ensemble; expected 5")
        elif len(selected) > 5:
            selected = selected[:5]

        # Log final composition and store selection
        sel_str = ", ".join([f"CV{e['cv']}-Fold{e['fold']} (Cat A={e['total_cat_a']}, wMAE={e['wmae']:.4f})" for e in selected])
        self._logger.info(f"Final ensemble composition: {sel_str}")
        # persist selected models for downstream analyses (ordinal per model)
        self.selected_models = [{k: e[k] for k in ['cv', 'fold', 'wmae', 'preds', 'idx', 'total_cat_a', 'property_cat_a'] if k in e} for e in selected]

        # Ensemble averaging
        top_preds = [e['preds'] for e in selected]
        preds_stack = np.stack(top_preds, axis=0)
        avg_preds = np.mean(preds_stack, axis=0)
        metrics = self._calculate_metrics(avg_preds, self.labels)
        metrics["num_models"] = len(selected)
        self.ensemble_results = metrics
        # Store ensemble stats for downstream analyses
        self.ensemble_preds_mean = avg_preds
        self.ensemble_preds_std = np.std(preds_stack, axis=0)
        self._logger.info(f"Ensemble RÂ²={metrics['r2']:.4f}, wMAE={metrics['wmae']:.4f}")


    def _evaluate_property_ensembles(self) -> None:
        """Build per-property ensembles using Category-A counts as primary criterion."""
        if not self.individual_results or not self.all_predictions:
            raise ValueError('No individual model results available for per-property ensembling')

        # Precompute ordinal scores
        ordinal_scores = []
        for i, preds in enumerate(self.all_predictions):
            score = self._compute_ordinal_score_for_model(preds)
            ordinal_scores.append(score)

        def _prop_mae_of_model(model_idx: int, prop: str) -> float:
            try:
                val = float(self.individual_results[model_idx].get(f'mae_{prop}', float('inf')))
                return val if np.isfinite(val) else float('inf')
            except Exception:
                return float('inf')

        for p_idx, prop in enumerate(self.property_names):
            entries = []
            for i, m in enumerate(self.individual_results):
                cat_a = ordinal_scores[i]['property_cat_a'].get(prop, 0)
                wmae = float(m.get('wmae', np.inf))
                prop_mae = _prop_mae_of_model(i, prop)
                entries.append({
                    'idx': i,
                    'cv': int(m.get('cv')) if 'cv' in m else m.get('cv'),
                    'fold': int(m.get('fold')) if 'fold' in m else None,
                    'cat_a': int(cat_a),
                    'prop_mae': prop_mae,
                    'wmae': wmae,
                    'preds': self.all_predictions[i],
                })

            from collections import defaultdict
            groups = defaultdict(list)
            for e in entries:
                groups[e['cv']].append(e)

            selected: List[dict] = []
            if self.prop_ens_enforce_cv:
                for required_cv in [1, 2, 3]:
                    if required_cv in groups and groups[required_cv]:
                        best = sorted(groups[required_cv], key=lambda x: (-x['cat_a'], x['prop_mae'], x['wmae']))[0]
                        selected.append(best)
                        self._logger.info(f"[{prop}] CV{required_cv} best: Fold{best['fold']} (Cat A={best['cat_a']}, MAE={best['prop_mae']:.4f}, wMAE={best['wmae']:.4f})")

            selected_indices = {e['idx'] for e in selected}
            remaining = [e for e in entries if e['idx'] not in selected_indices]
            remaining_sorted = sorted(remaining, key=lambda x: (-x['cat_a'], x['prop_mae'], x['wmae']))
            for e in remaining_sorted:
                if len(selected) >= self.prop_ens_k:
                    break
                selected.append(e)

            if not selected:
                self._logger.warning(f"[{prop}] No models available for per-property ensemble; skipping")
                continue

            self.selected_models_by_property[prop] = [{
                'cv': s['cv'],
                'fold': s['fold'],
                'cat_a': s['cat_a'],
                'prop_mae': float(s['prop_mae']) if np.isfinite(s['prop_mae']) else float('nan'),
                'wmae': float(s['wmae']) if np.isfinite(s['wmae']) else float('nan')
            } for s in selected]

            cols = []
            for s in selected:
                preds = s['preds']
                if preds is None:
                    continue
                cols.append(np.asarray(preds)[:, p_idx])
            if not cols:
                self._logger.warning(f"[{prop}] No predictions collected for selected models; skipping")
                continue
            stack = np.stack(cols, axis=0)
            y_pred = np.mean(stack, axis=0)
            self.ensemble_preds_mean_by_property[prop] = y_pred

            y_true = self.labels[:, p_idx] if self.labels is not None else None
            if y_true is None:
                self._logger.warning(f"[{prop}] Labels not available; metrics skipped")
                continue
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            n = int(mask.sum())
            if n == 0:
                self._logger.warning(f"[{prop}] No valid samples for metrics; skipping")
                continue
            y_t = y_true[mask]
            y_p = y_pred[mask]
            mae = float(mean_absolute_error(y_t, y_p))
            r2 = float(self._r2(y_t, y_p))
            nrmse_iqr = float(self._nrmse_iqr(y_t, y_p))
            self.ensemble_results_by_property[prop] = {
                'n': n,
                'mae': mae,
                'r2': r2,
                'nrmse_iqr': nrmse_iqr,
                'num_models': len(selected)
            }
            self._logger.info(f"[{prop}] Ensemble: MAE={mae:.4f}, R2={r2:.4f}, NRMSE_IQR={nrmse_iqr:.4f}, K={len(selected)}")

        # Write JSONs + CSV summary
        try:
            with open(os.path.join(self.eval_dir, 'per_property_ensemble_selection.json'), 'w') as f:
                json.dump(self.selected_models_by_property, f, indent=2)
            with open(os.path.join(self.eval_dir, 'per_property_ensemble_metrics.json'), 'w') as f:
                json.dump(self.ensemble_results_by_property, f, indent=2)
            # CSV metrics summary
            if self.ensemble_results_by_property:
                rows = []
                for prop, met in self.ensemble_results_by_property.items():
                    rows.append({
                        'property': prop,
                        'n': met.get('n', 0),
                        'mae': met.get('mae', float('nan')),
                        'r2': met.get('r2', float('nan')),
                        'nrmse_iqr': met.get('nrmse_iqr', float('nan')),
                        'num_models': met.get('num_models', 0),
                    })
                pd.DataFrame(rows).to_csv(os.path.join(self.eval_dir, 'per_property_ensemble_metrics.csv'), index=False)
        except Exception as e:
            self._logger.warning(f"Failed to write per-property ensemble artifacts: {e}")
    def _select_best_models(self) -> List[Dict]:
        """Build info about selected ensemble models including ordinal scores."""
        best_models = []
        if self.selected_models:
            for sm in self.selected_models:
                # Find corresponding individual result to get all metrics
                idx = sm.get('idx')
                if idx is not None and idx < len(self.individual_results):
                    m = self.individual_results[idx]
                    info = {
                        "cv": sm.get("cv"),
                        "fold": sm.get("fold"),
                        "metrics": {k: float(m[k]) for k in m if k.startswith("mae_") or k == "wmae"},
                        "ordinal_scores": {
                            "total_cat_a": sm.get("total_cat_a", 0),
                            "property_cat_a": sm.get("property_cat_a", {}),
                        },
                    }
                    best_models.append(info)
        else:
            # Fallback: use top 5 by wMAE if selected_models not available
            sorted_models = sorted(self.individual_results, key=lambda x: x["wmae"])[:5]
            for m in sorted_models:
                info = {
                    "cv": m["cv"],
                    "fold": m["fold"],
                    "metrics": {k: float(m[k]) for k in m if k.startswith("mae_") or k == "wmae"},
                }
                best_models.append(info)
        return best_models

    def _save_best_models(self):
        best_models = self._select_best_models()
        path = os.path.join(self.eval_dir, "best_models.json")
        with open(path, "w") as f:
            json.dump({
                "models": best_models,
                "selection_method": "ordinal_analysis_guided",
                "description": "Models selected based on Category A performance (highest precision predictions) with at least 1 from each CV"
            }, f, indent=4)
        self._logger.info(f"Saved best models to {path}")

    def _save_results(self):
        df = pd.DataFrame(self.individual_results)
        df.to_csv(os.path.join(self.eval_dir, "individual_results.csv"), index=False)
        with open(os.path.join(self.eval_dir, "ensemble_results.json"), "w") as f:
            json.dump(self.ensemble_results, f, indent=4)

    def _generate_summary_report(self):
        r2_values = [r["r2"] for r in self.individual_results]
        wmae_values = [r["wmae"] for r in self.individual_results]
        report_lines = [
            "Final Evaluation Summary",
            "=" * 30,
            f"Mean RÂ²: {np.mean(r2_values):.4f}",
            f"Std RÂ²: {np.std(r2_values):.4f}",
            f"Ensemble RÂ²: {self.ensemble_results['r2']:.4f}",
            "",
            f"Mean wMAE: {np.mean(wmae_values):.4f}",
            f"Std wMAE: {np.std(wmae_values):.4f}",
            f"Ensemble wMAE: {self.ensemble_results['wmae']:.4f}",
        ]
        # Add CV artifact summary to aid stacking/SHAP troubleshooting
        try:
            cv_dir = self._find_latest_cv_dir()
            if cv_dir is not None:
                oof_parquet = os.path.join(cv_dir, 'oof_predictions.parquet')
                oof_csv = os.path.join(cv_dir, 'oof_predictions.csv')
                emb_path = os.path.join(cv_dir, 'embeddings.npy')
                def _flag(p: str) -> str:
                    return 'present' if os.path.exists(p) else 'missing'
                report_lines.extend([
                    "",
                    "Selected CV Directory:",
                    cv_dir,
                    f"- oof_predictions.parquet: {_flag(oof_parquet)}",
                    f"- oof_predictions.csv: {_flag(oof_csv)}",
                    f"- embeddings.npy: {_flag(emb_path)}",
                ])
            else:
                report_lines.extend(["", "Selected CV Directory:", "(none found)"])
        except Exception:
            # Do not fail summary if filesystem checks error out
            pass
        with open(os.path.join(self.eval_dir, "summary_report.txt"), "w") as f:
            f.write("\n".join(report_lines))

    def evaluate_all(self):
        self._evaluate_individual_models()
        self._evaluate_ensemble()
        # Build per-property ensembles using ordinal Category-A
        try:
            self._evaluate_property_ensembles()
        except Exception as e:
            self._logger.warning(f"Per-property ensemble selection failed: {e}")
        self._save_best_models()
        self._save_results()
        self._generate_summary_report()
        # Perform stacking and SHAP artifacts
        if getattr(self.config, 'STACKING_ENABLED', True):
            self._run_stacking_and_shap()
        # Additive: ordinal analysis per property
        try:
            self._run_ordinal_analysis_clean()
        except Exception as e:
            self._logger.warning(f"Ordinal analysis failed: {e}")
        # Optional: report calibrated FFV test metrics if calibrator exists
        try:
            self._report_ffv_test_calibrated_metrics()
        except Exception as e:
            self._logger.debug(f'FFV test calibration reporting skipped: {e}')

    # ---------------------
    # Stacking and SHAP
    # ---------------------
    def _find_latest_cv_dir(self) -> Optional[str]:
        if not os.path.isdir(self.cv_dir):
            return None
        cvs = [d for d in os.listdir(self.cv_dir) if d.startswith('cv') and os.path.isdir(os.path.join(self.cv_dir, d))]
        if not cvs:
            return None
        latest = sorted(cvs, key=lambda s: int(s[2:]))[-1]
        return os.path.join(self.cv_dir, latest)

    def _read_oof(self, cv_run_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
        oof_path_parquet = os.path.join(cv_run_dir, 'oof_predictions.parquet')
        oof_path_csv = os.path.join(cv_run_dir, 'oof_predictions.csv')
        if os.path.exists(oof_path_parquet):
            oof_df = pd.read_parquet(oof_path_parquet)
        elif os.path.exists(oof_path_csv):
            oof_df = pd.read_csv(oof_path_csv)
        else:
            raise FileNotFoundError('OOF predictions not found in latest CV dir')
        emb_path = os.path.join(cv_run_dir, 'embeddings.npy')
        if not os.path.exists(emb_path):
            raise FileNotFoundError('embeddings.npy not found in latest CV dir')
        emb = np.load(emb_path)
        return oof_df, emb

    def _log_cv_artifacts(self, cv_run_dir: str) -> None:
        oof_parquet = os.path.join(cv_run_dir, 'oof_predictions.parquet')
        oof_csv = os.path.join(cv_run_dir, 'oof_predictions.csv')
        emb_path = os.path.join(cv_run_dir, 'embeddings.npy')
        def _fmt(path: str) -> str:
            if os.path.exists(path):
                try:
                    size = os.path.getsize(path)
                    return f"present ({size} bytes)"
                except Exception:
                    return "present"
            return "missing"
        self._logger.info(f"Using CV dir: {cv_run_dir}")
        self._logger.info(f"  oof_predictions.parquet: {_fmt(oof_parquet)}")
        self._logger.info(f"  oof_predictions.csv: {_fmt(oof_csv)}")
        self._logger.info(f"  embeddings.npy: {_fmt(emb_path)}")

    # ---------------------
    # FFV Isotonic calibration helpers
    # ---------------------
    @staticmethod
    def _nrmse_iqr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        q75, q25 = np.percentile(y_true, [75, 25])
        iqr = max(1e-12, (q75 - q25))
        return rmse / iqr

    @staticmethod
    def _affine_fit(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        X = np.vstack([y_pred, np.ones_like(y_pred)]).T
        a, b = np.linalg.lstsq(X, y_true, rcond=None)[0]
        return float(a), float(b)

    @staticmethod
    def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2:
            return float('nan')
        return float(np.corrcoef(y_true, y_pred)[0, 1])

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot <= 1e-12:
            return float('nan') if ss_res > 0 else 1.0
        return 1.0 - float(ss_res / ss_tot)

    def _map_oof_columns_ffv(self, oof_df: pd.DataFrame) -> pd.DataFrame:
        df = oof_df.copy()
        # fold column mapping
        for c in ['fold', 'cv_fold', 'Fold', 'kfold']:
            if c in df.columns:
                df = df.rename(columns={c: 'fold'})
                break
        # true label mapping
        for c in ['FFV_label', 'FFV_true', 'true_FFV', 'label_FFV']:
            if c in df.columns:
                df = df.rename(columns={c: 'FFV_true'})
                break
        # prediction mapping
        for c in ['FFV_oof', 'FFV_pred', 'pred_FFV']:
            if c in df.columns:
                df = df.rename(columns={c: 'FFV_pred'})
                break
        if not {'fold', 'FFV_true', 'FFV_pred'}.issubset(set(df.columns)):
            missing = {'fold', 'FFV_true', 'FFV_pred'} - set(df.columns)
            raise ValueError(f"OOF file missing required columns for FFV calibration: {missing}")
        return df

    # --- New generic OOF mapping/join utilities for stacking & calibration ---
    def _map_oof_columns_generic(self, oof_df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Standardize OOF dataframe to have columns: 'fold', 'id', f'{target}_oof'.
        Does not enforce label presence (joined separately if needed)."""
        df = oof_df.copy()
        # fold mapping
        for c in ['fold', 'cv_fold', 'Fold', 'kfold']:
            if c in df.columns:
                df = df.rename(columns={c: 'fold'})
                break
        # id mapping
        if 'id' not in df.columns:
            for c in ['ID', 'compound_id', 'molecule_id', self.config.compound_id_name]:
                if c in df.columns:
                    df = df.rename(columns={c: 'id'})
                    break
        # oof pred mapping for target
        tgt_oof = f'{target}_oof'
        if tgt_oof not in df.columns:
            for c in [f'{target}_pred', f'pred_{target}', f'oof_{target}']:
                if c in df.columns:
                    df = df.rename(columns={c: tgt_oof})
                    break
        # ensure required minimums exist
        for req in ['fold', 'id', tgt_oof]:
            if req not in df.columns:
                self._logger.debug(f"_map_oof_columns_generic: missing '{req}' for target {target}")
        return df

    def _join_oof_with_labels(self, oof_df: pd.DataFrame, target: str) -> pd.DataFrame:
        # map columns for OOF preds and true label
        df = self._map_oof_columns_generic(oof_df, target)
        # ensure label column presence: prefer in-OOF labels first
        candidate_label_cols = [f'{target}_label', f'{target}_true', f'true_{target}', f'label_{target}', target]
        found_label = None
        for c in candidate_label_cols:
            if c in df.columns:
                found_label = c
                break
        if found_label is not None and found_label != target:
            df = df.rename(columns={found_label: target})
        elif found_label is None:
            # merge labels from dataset meta if needed and available
            try:
                meta_df = getattr(self.data_module, 'dataset', None)
                meta_df = meta_df.meta if meta_df is not None else None
                if meta_df is not None and target in meta_df.columns:
                    y = meta_df[['id', target]].copy()
                    df = df.merge(y, on='id', how='left')
                else:
                    self._logger.debug(f"Meta labels unavailable for {target}")
            except Exception as e:
                self._logger.debug(f"Label merge from meta failed for {target}: {e}")

        label_col = target
        # drop NaNs safely for both label and oof
        if label_col in df.columns:
            df = df[df[label_col].notna()]
        oof_col = f"{target}_oof"
        if oof_col in df.columns:
            df = df[df[oof_col].notna()]
        return df

    def _fit_ffv_isotonic(self, oof_df: pd.DataFrame):
        if IsotonicRegression is None or not getattr(self.config, 'CALIBRATE_FFV', True):
            return None
        df = self._join_oof_with_labels(oof_df, 'FFV')
        if len(df) < 20:
            self._logger.warning("FFV calibration skipped: not enough labeled rows")
            return None
        iso = IsotonicRegression(out_of_bounds='clip')
        try:
            iso.fit(df['FFV_oof'].to_numpy(), df['FFV'].to_numpy())
        except Exception as e:
            self._logger.warning(f"FFV isotonic fit failed: {e}")
            return None
        return iso

    def _save_ffv_calibrator(self, calibrator) -> None:
        try:
            calib_dir = Path(self.config.output_dir) / str(getattr(self.config, 'CALIBRATION_DIR', getattr(self.config, 'CALIBRATION_OUTDIR', 'calibration/'))).strip('/')
            calib_dir.mkdir(parents=True, exist_ok=True)
            if joblib is not None:
                joblib.dump(calibrator, calib_dir / 'ffv_isotonic.pkl')
            else:
                # lightweight fallback serialization
                X = getattr(calibrator, 'X_thresholds_', None)
                Y = getattr(calibrator, 'y_thresholds_', None)
                if X is not None and Y is not None:
                    np.savez(calib_dir / 'ffv_isotonic_fallback.npz', X=X, Y=Y)
            self._logger.info(f"Saved FFV calibrator to {calib_dir}")
        except Exception as e:
            self._logger.warning(f"Failed to save FFV calibrator: {e}")

    def _compute_base_oof_mae(self, oof_df: pd.DataFrame) -> Dict[str, float]:
        maes: Dict[str, float] = {}
        for p in self.property_names:
            y = oof_df.get(f'{p}_label', None)
            yhat = oof_df.get(f'{p}_oof', None)
            if y is None or yhat is None:
                continue
            mask = np.isfinite(y) & np.isfinite(yhat)
            if mask.any():
                maes[p] = float(mean_absolute_error(y[mask], yhat[mask]))
        return maes

    def _build_stacker_matrix(self, oof_df: pd.DataFrame, emb: np.ndarray, df_join: pd.DataFrame, target: str):
        # identify OOF columns
        oof_cols = [c for c in df_join.columns if c.endswith('_oof')]
        # simplest: slice embeddings to df_join.index order
        E = emb[df_join.index.to_numpy(), :]
        O = df_join[oof_cols].to_numpy()
        Z = np.hstack([O, E])
        return Z, oof_cols

    def _train_stacker_and_shap(self, target: str, oof_df: pd.DataFrame, emb: np.ndarray):
        import numpy as _np
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        try:
            import shap
        except Exception:
            shap = None
        try:
            df_join = self._join_oof_with_labels(oof_df, target)
            if len(df_join) < 30:
                self._logger.warning(f"{target} stacker skipped: insufficient labeled rows ({len(df_join)})")
                return None
            Z, oof_cols = self._build_stacker_matrix(oof_df, emb, df_join, target)
            y = df_join[target].to_numpy()
            # choose PCA size to avoid p>>n
            n = Z.shape[0]
            pca_k = max(8, min(32, n // 5 if n >= 5 else 1))
            n_components = max(1, min(pca_k, Z.shape[1], n))
            stacker = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components)),
                ('ridge', Ridge(alpha=1.0, random_state=0))
            ])
            stacker.fit(Z, y)
            # Save model
            model_path = os.path.join(self.stack_dir, f"{target}_stacker.pkl")
            if joblib is not None:
                joblib.dump(stacker, model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(stacker, f)
            # SHAP on PCA-transformed space (LinearExplainer)
            if shap is not None:
                try:
                    Zp = stacker.named_steps['pca'].transform(Z)
                    explainer = shap.LinearExplainer(stacker.named_steps['ridge'], Zp, feature_perturbation='interventional')
                    shap_vals = explainer.shap_values(Zp)
                except Exception as e:
                    raise RuntimeError(f"LinearExplainer failed: {e}")
                # Persist artifacts (save at root of stack_shap_dir per acceptance)
                os.makedirs(self.stack_shap_dir, exist_ok=True)
                try:
                    import matplotlib.pyplot as plt
                    shap.summary_plot(shap_vals, Zp, show=False)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.stack_shap_dir, f'{target}_beeswarm.png'), dpi=200)
                    plt.close()
                    shap.summary_plot(shap_vals, Zp, plot_type='bar', show=False)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.stack_shap_dir, f'{target}_summary_bar.png'), dpi=200)
                    plt.close()
                except Exception as e:
                    self._logger.warning(f"{target} SHAP plotting failed: {e}")
                # CSV export (ensure numeric)
                sv = _np.array(shap_vals, dtype=float)
                pd.DataFrame(sv).to_csv(os.path.join(self.stack_shap_dir, f'{target}_shap_values.csv'), index=False)
            else:
                # Fallback: permutation importance CSV
                self._logger.warning("shap not installed; saving permutation importance CSV as fallback")
                try:
                    from sklearn.inspection import permutation_importance
                    pi = permutation_importance(stacker, Z, y, n_repeats=10, random_state=0)
                    os.makedirs(self.stack_shap_dir, exist_ok=True)
                    pd.DataFrame({'feature': np.arange(Z.shape[1]), 'importance_mean': pi.importances_mean, 'importance_std': pi.importances_std}).to_csv(
                        os.path.join(self.stack_shap_dir, f'{target}_perm_importance.csv'), index=False)
                except Exception as e:
                    self._logger.warning(f"{target} permutation importance fallback failed: {e}")
            return stacker
        except Exception as e:
            self._logger.warning(f"{target} PCARidge SHAP failed: {e}. Falling back to LightGBM + TreeSHAP.")
            try:
                import lightgbm as lgb
                try:
                    import shap
                except Exception:
                    shap = None
                df_join = self._join_oof_with_labels(oof_df, target)
                Z, oof_cols = self._build_stacker_matrix(oof_df, emb, df_join, target)
                y = df_join[target].to_numpy()
                # reduce feature count by simple subsampling if needed
                max_feats = min(Z.shape[1], max(64, Z.shape[0] // 2))
                Zr = Z[:, :max_feats]
                model = lgb.LGBMRegressor(
                    n_estimators=400, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=0
                )
                model.fit(Zr, y)
                if joblib is not None:
                    joblib.dump(model, os.path.join(self.stack_dir, f"{target}_stacker.pkl"))
                # TreeSHAP
                if shap is not None:
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(Zr)
                        import matplotlib.pyplot as plt
                        shap.summary_plot(shap_vals, Zr, show=False)
                        plt.tight_layout(); plt.savefig(os.path.join(self.stack_shap_dir, f'{target}_beeswarm.png'), dpi=200); plt.close()
                        shap.summary_plot(shap_vals, Zr, plot_type='bar', show=False)
                        plt.tight_layout(); plt.savefig(os.path.join(self.stack_shap_dir, f'{target}_summary_bar.png'), dpi=200); plt.close()
                        pd.DataFrame(np.array(shap_vals, dtype=float)).to_csv(os.path.join(self.stack_shap_dir, f'{target}_shap_values.csv'), index=False)
                    except Exception as e2:
                        self._logger.warning(f"{target} TreeSHAP plotting failed: {e2}")
                else:
                    self._logger.warning("shap not installed; skipping TreeSHAP artifact generation")
                return model
            except Exception as e2:
                # absolute fallback: permutation importance
                self._logger.warning(f"{target} TreeSHAP failed: {e2}. Writing permutation importance.")
                try:
                    from sklearn.inspection import permutation_importance
                    from sklearn.linear_model import Ridge
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import Pipeline
                    df_join = self._join_oof_with_labels(oof_df, target)
                    Z, _ = self._build_stacker_matrix(oof_df, emb, df_join, target)
                    y = df_join[target].to_numpy()
                    # small ridge
                    model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]).fit(Z, y)
                    pi = permutation_importance(model, Z, y, n_repeats=10, random_state=0)
                    os.makedirs(self.stack_shap_dir, exist_ok=True)
                    pd.DataFrame({'feature': np.arange(Z.shape[1]), 'importance_mean': pi.importances_mean, 'importance_std': pi.importances_std}).to_csv(
                        os.path.join(self.stack_shap_dir, f'{target}_perm_importance.csv'), index=False)
                except Exception as e3:
                    self._logger.warning(f"{target} permutation importance failed: {e3}")
                return None

    def _run_stacking_and_shap(self) -> None:
        self._logger.info('Running stacking + SHAP...')
        cv_run_dir = self._find_latest_cv_dir()
        if cv_run_dir is None:
            self._logger.warning('No CV directory found; skipping stacking')
            return
        # Log which CV dir is selected and whether required artifacts exist
        try:
            self._log_cv_artifacts(cv_run_dir)
        except Exception as e:
            self._logger.debug(f'Failed to log CV artifacts: {e}')
        try:
            oof_df, emb = self._read_oof(cv_run_dir)
        except FileNotFoundError as e:
            self._logger.warning(f'OOF artifacts missing in {cv_run_dir}; skipping stacking/SHAP. Reason: {e}')
            return
        # Safe FFV isotonic calibration on valid rows only
        try:
            iso = self._fit_ffv_isotonic(oof_df)
            if iso is not None:
                self._save_ffv_calibrator(iso)
        except Exception as e:
            self._logger.warning(f"FFV calibration skipped: {e}")

        # Train stackers and write SHAP; prioritize FFV
        from sklearn.metrics import mean_absolute_error as _mae
        base_mae_report = {}
        report: Dict[str, Dict[str, float]] = {}
        targets_for_shap = ['FFV']
        for t in targets_for_shap:
            model = self._train_stacker_and_shap(t, oof_df, emb)
            # compute base vs stacker OOF MAE for t
            try:
                dfj = self._join_oof_with_labels(oof_df, t)
                base_mae = _mae(dfj[t], dfj[f"{t}_oof"]) if len(dfj) else float('nan')
                if model is not None:
                    Z, _ = self._build_stacker_matrix(oof_df, emb, dfj, t)
                    pred = model.predict(Z)
                    stack_mae = _mae(dfj[t], pred)
                else:
                    stack_mae = float('nan')
                report[t] = {'base_oof_mae': float(base_mae), 'stack_oof_mae': float(stack_mae), 'delta_mae': float(stack_mae - base_mae) if np.isfinite(base_mae) and np.isfinite(stack_mae) else float('nan')}
            except Exception as e:
                self._logger.warning(f"{t} stacker report failed: {e}")

        # Save stack report next to other eval artifacts
        try:
            with open(os.path.join(self.eval_dir, 'stack_report.json'), 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            self._logger.warning(f"Failed to write stack_report.json: {e}")

        # Optional Tg calibration
        if getattr(self.config, 'CALIBRATE_TG', False):
            try:
                self._calibrate_tg(oof_df)
            except Exception as e:
                self._logger.warning(f'Tg calibration failed: {e}')

    # (Replaced by per-target PCARidge/LightGBM SHAP in _train_stacker_and_shap)

    def _calibrate_tg(self, oof_df: pd.DataFrame) -> None:
        out_dir = os.path.join(self.config.output_dir, self.config.CALIBRATION_OUTDIR.strip('/'))
        os.makedirs(out_dir, exist_ok=True)
        target = 'Tg'
        y = oof_df.get(f'{target}_label')
        yhat = oof_df.get(f'{target}_oof')
        if y is None or yhat is None:
            self._logger.warning('No Tg OOF available for calibration')
            return
        mask = np.isfinite(y) & np.isfinite(yhat)
        if not mask.any():
            self._logger.warning('No valid Tg rows for calibration')
            return
        y = np.asarray(y[mask])
        yhat = np.asarray(yhat[mask])
        # Quantile mapping
        qs = np.linspace(0, 1, 51)
        pred_q = np.quantile(yhat, qs)
        true_q = np.quantile(y, qs)
        # Evaluate before/after on OOF
        def apply_map(v):
            return np.interp(v, pred_q, true_q)
        yhat_cal = apply_map(yhat)
        report = {
            'before_mae': float(mean_absolute_error(y, yhat)),
            'after_mae': float(mean_absolute_error(y, yhat_cal))
        }
        np.savez(os.path.join(out_dir, 'tg_quantile_map.npz'), pred_q=pred_q, true_q=true_q)
        with open(os.path.join(out_dir, 'tg_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

    def _run_ordinal_analysis_clean(self) -> None:
        """Clean, ASCII-safe ordinal analysis for ensemble and selected models."""
        if self.ensemble_preds_mean is None or self.labels is None:
            self._logger.warning('Ensemble predictions not ready; skipping ordinal analysis')
            return
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import mean_squared_error, r2_score
        except Exception as e:
            self._logger.warning(f'Missing plotting/metrics deps: {e}; skipping ordinal analysis')
            return

        # Resolve IDs for test set
        meta_path = self.config.get_processed_file_path('meta', 'primary')
        if os.path.exists(meta_path):
            meta_df = pd.read_csv(meta_path)
            test_meta = meta_df[meta_df['group'] == 'test'].reset_index(drop=True)
            if len(test_meta) == self.labels.shape[0]:
                id_col = self.config.compound_id_name if self.config.compound_id_name in test_meta.columns else test_meta.columns[0]
                ids = test_meta[id_col].astype(str).tolist()
            else:
                ids = [f'test_{i}' for i in range(self.labels.shape[0])]
                self._logger.warning('Test meta size mismatch; using synthetic IDs')
        else:
            ids = [f'test_{i}' for i in range(self.labels.shape[0])]
            self._logger.warning('Meta file missing; using synthetic IDs for ordinal analysis')

        base_out_ens = os.path.join(self.eval_dir, 'ordinal_analysis')
        os.makedirs(base_out_ens, exist_ok=True)
        base_out_models = os.path.join(self.eval_dir, 'ordinal_analysis_models')
        os.makedirs(base_out_models, exist_ok=True)

        def assign_levels(abs_err: np.ndarray, thrA: float, thrB: float, thrC: float) -> np.ndarray:
            levels = np.full_like(abs_err, 'D', dtype=object)
            levels[abs_err <= thrC] = 'C'
            levels[abs_err <= thrB] = 'B'
            levels[abs_err <= thrA] = 'A'
            return levels

        def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
            if y_true.size == 0:
                return {'Total Samples': 0.0, 'MAE': float('nan'), 'MSE': float('nan'), 'RMSE': float('nan'), 'R2': float('nan'), 'Correlation': float('nan')}
            mae = float(mean_absolute_error(y_true, y_pred))
            mse = float(mean_squared_error(y_true, y_pred))
            rmse = float(np.sqrt(mse))
            # r2_score undefined for fewer than 2 samples; guard to avoid warnings
            if y_true.size >= 2:
                r2 = float(r2_score(y_true, y_pred))
            else:
                r2 = float('nan')
                try:
                    self._logger.info(f"R2 skipped in ordinal analysis: insufficient samples (n={y_true.size})")
                except Exception:
                    pass
            try:
                corr = float(np.corrcoef(y_true, y_pred)[0, 1])
            except Exception:
                corr = float('nan')
            return {'Total Samples': float(y_true.size), 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Correlation': corr}

        summary_rows: List[Dict[str, float]] = []

        def analyze_matrix(pred_mat: np.ndarray, std_mat: Optional[np.ndarray], out_root: str, model_label: str, cv: str = '', fold: str = '') -> None:
            for p_idx, prop in enumerate(self.property_names):
                y_true = self.labels[:, p_idx]
                y_pred = pred_mat[:, p_idx]
                y_std = std_mat[:, p_idx] if std_mat is not None else np.full_like(y_pred, np.nan)
                mask = np.isfinite(y_true) & np.isfinite(y_pred)
                if not mask.any():
                    continue
                y_t, y_p, y_s = y_true[mask], y_pred[mask], y_std[mask]
                id_arr = np.array(ids)[mask]
                # thresholds
                rng = getattr(self.config, 'property_ranges', {}).get(prop) if hasattr(self.config, 'property_ranges') else None
                if rng is not None:
                    delta = float(rng[1] - rng[0])
                    thrA, thrB, thrC = 0.05 * delta, 0.10 * delta, 0.20 * delta
                else:
                    abs_err_all = np.abs(y_p - y_t)
                    thrA, thrB, thrC = np.quantile(abs_err_all, [0.25, 0.50, 0.75])
                abs_err = np.abs(y_p - y_t)
                levels = assign_levels(abs_err, thrA, thrB, thrC)

                out_dir = os.path.join(out_root, prop)
                os.makedirs(out_dir, exist_ok=True)
                df = pd.DataFrame({'ID': id_arr, 'true': y_t, 'pred': y_p, 'pred_std': y_s, 'abs_error': abs_err, 'confidence_level': levels})
                df.to_csv(os.path.join(out_dir, f'{prop}_ordinal_table.csv'), index=False)

                counts = df['confidence_level'].value_counts().reindex(['A', 'B', 'C', 'D']).fillna(0).astype(int)
                with open(os.path.join(out_dir, f'{prop}_distribution.json'), 'w') as f:
                    json.dump({'total': int(counts.sum()), 'thresholds': {'A': float(thrA), 'B': float(thrB), 'C': float(thrC)}, 'counts': {k: int(v) for k, v in counts.to_dict().items()}}, f, indent=2)

                # Plots
                conf_colors = {'A': '#2ca02c', 'B': '#1f77b4', 'C': '#ff7f0e', 'D': '#d62728'}
                min_val = float(min(np.min(y_t), np.min(y_p)))
                max_val = float(max(np.max(y_t), np.max(y_p)))
                pad = (max_val - min_val) * 0.05
                for level in ['A', 'B', 'C', 'D']:
                    sub = df[df['confidence_level'] == level]
                    if sub.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.scatter(sub['pred'], sub['true'], alpha=0.7, color=conf_colors[level], s=50, label=f'n = {len(sub)}')
                    ax.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], linestyle='--', color='black', lw=2)
                    m = calc_metrics(sub['true'].values, sub['pred'].values)
                    ax.text(0.05, 0.95, f"R2: {m['R2']:.3f}\nRMSE: {m['RMSE']:.3f}\nMAE: {m['MAE']:.3f}", transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
                    ax.set_title(f"{prop} - Level {level} (n = {len(sub)})")
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.set_xlim(min_val - pad, max_val + pad)
                    ax.set_ylim(min_val - pad, max_val + pad)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f'confidence_level_{level}.png'), dpi=300, bbox_inches='tight')
                    plt.close(fig)

                fig, ax = plt.subplots(figsize=(10, 10))
                for level in ['A', 'B', 'C', 'D']:
                    sub = df[df['confidence_level'] == level]
                    if sub.empty:
                        continue
                    ax.scatter(sub['pred'], sub['true'], alpha=0.7, color=conf_colors[level], s=50, label=f'Level {level} (n = {len(sub)})')
                ax.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], linestyle='--', color='black', lw=2, label='Perfect Prediction')
                overall = calc_metrics(df['true'].values, df['pred'].values)
                ax.text(0.05, 0.95, f"Overall\nR2: {overall['R2']:.3f}\nRMSE: {overall['RMSE']:.3f}\nMAE: {overall['MAE']:.3f}\nCorr: {overall['Correlation']:.3f}", transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
                ax.set_title(f"{prop} - Overall Prediction Performance")
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_xlim(min_val - pad, max_val + pad)
                ax.set_ylim(min_val - pad, max_val + pad)
                ax.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'overall_analysis.png'), dpi=300, bbox_inches='tight')
                plt.close(fig)

                # Metrics CSV per property
                rows = {lvl: calc_metrics(df[df['confidence_level'] == lvl]['true'].values, df[df['confidence_level'] == lvl]['pred'].values) for lvl in ['A', 'B', 'C', 'D']}
                rows['Overall'] = overall
                pd.DataFrame(rows).T.to_csv(os.path.join(out_dir, 'confidence_metrics.csv'))

                # Add to summary
                summary_rows.append({
                    'model': model_label, 'cv': cv, 'fold': fold, 'property': prop,
                    'A': int(counts.get('A', 0)), 'B': int(counts.get('B', 0)), 'C': int(counts.get('C', 0)), 'D': int(counts.get('D', 0)), 'total': int(counts.sum()),
                    'thrA': float(thrA), 'thrB': float(thrB), 'thrC': float(thrC),
                    'MAE': float(np.mean(np.abs(df['true'] - df['pred']))), 'RMSE': float(np.sqrt(np.mean((df['true'] - df['pred'])**2))),
                    'R2': float(overall['R2']), 'Correlation': float(overall['Correlation']),
                })

        # Ensemble
        analyze_matrix(self.ensemble_preds_mean, self.ensemble_preds_std, base_out_ens, 'Ensemble', '', '')
        # Selected models
        if self.selected_models:
            for sm in self.selected_models:
                preds = sm.get('preds')
                if preds is None:
                    continue
                cv = f"{sm.get('cv')}" if sm.get('cv') is not None else ''
                fold = f"{sm.get('fold')}" if sm.get('fold') is not None else ''
                out_root = os.path.join(base_out_models, f"CV{cv}_Fold{fold}")
                os.makedirs(out_root, exist_ok=True)
                analyze_matrix(np.asarray(preds), None, out_root, f"CV{cv}_Fold{fold}", cv, fold)

        # Save compact comparison
        if summary_rows:
            comp_path = os.path.join(self.eval_dir, 'ordinal_comparison_summary.csv')
            pd.DataFrame(summary_rows).to_csv(comp_path, index=False)
            self._logger.info(f'Wrote ordinal comparison summary: {comp_path}')

    def _load_ffv_calibrator(self):
        if not getattr(self.config, 'CALIBRATE_FFV', True):
            return None
        calib_dir = Path(self.config.output_dir) / str(getattr(self.config, 'CALIBRATION_DIR', getattr(self.config, 'CALIBRATION_OUTDIR', 'calibration/'))).strip('/')
        pkl = calib_dir / 'ffv_isotonic.pkl'
        npz = calib_dir / 'ffv_isotonic_fallback.npz'
        try:
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
        except Exception:
            return None
        return None

    def _report_ffv_test_calibrated_metrics(self) -> None:
        if self.ensemble_preds_mean is None or self.labels is None:
            return
        # Find FFV index
        if 'FFV' not in self.property_names:
            return
        idx = self.property_names.index('FFV')
        y_true = self.labels[:, idx]
        y_pred = self.ensemble_preds_mean[:, idx]
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not mask.any():
            return
        y_t = y_true[mask]
        y_p = y_pred[mask]
        mae_raw = float(mean_absolute_error(y_t, y_p))
        calibrator = self._load_ffv_calibrator()
        out_dir = Path(self.config.output_dir) / str(getattr(self.config, 'CALIBRATION_DIR', getattr(self.config, 'CALIBRATION_OUTDIR', 'calibration/'))).strip('/')
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {'n': int(len(y_t)), 'mae_raw': mae_raw}
        if calibrator is not None:
            y_pc = calibrator.predict(y_p)
            report.update({
                'mae_cal': float(mean_absolute_error(y_t, y_pc)),
                'r2_raw': float(self._r2(y_t, y_p)),
                'r2_cal': float(self._r2(y_t, y_pc)),
                'nrmse_iqr_raw': float(self._nrmse_iqr(y_t, y_p)),
                'nrmse_iqr_cal': float(self._nrmse_iqr(y_t, y_pc)),
            })
        with open(out_dir / 'ffv_test_calibrated_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    cfg = Configuration()
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = FinalEvaluator(cfg)
    evaluator.evaluate_all()
