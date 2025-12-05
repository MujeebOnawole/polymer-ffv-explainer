# stat_val_64rel.py - Standalone statistical validation for 64-relation RGCN

import os
import sys
import json
import signal
import torch
import numpy as np
import pandas as pd
import csv
import glob
import torch.nn as nn
import gc
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, matthews_corrcoef
)

from model import BaseGNN
from data_module import MoleculeDataModule, collate_fn
from logger import LoggerSetup, get_logger
from config import Configuration
import psutil
import time
import shutil
import copy
import traceback
import logging
import random


# Initialize logger at the module level
logger = get_logger(__name__)

# Set PyTorch global settings upfront
torch.set_float32_matmul_precision('medium')
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def make_serializable(obj):
    """Convert objects to JSON serializable format."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return make_serializable(obj.cpu().tolist())
    else:
        return obj


def _fmt_metric_value(val, max_len: int = 8) -> str:
    """
    Pretty-format metrics that might be float, numpy scalar, or 1D arrays.
    - Scalars -> '0.1234'
    - 1D arrays -> '[0.1234, 0.5678, …]' (capped at max_len elements)
    - Higher-rank -> 'shape=(...)'
    - Fallback -> str(val)
    """
    import numpy as _np
    try:
        arr = _np.asarray(val)
    except Exception:
        return str(val)
    if arr.size == 1:
        try:
            return f"{float(arr):.4f}"
        except Exception:
            return str(arr.item() if arr.shape == () else arr)
    if arr.ndim == 1:
        n = arr.size
        head = ", ".join(f"{float(x):.4f}" for x in arr[:max_len])
        tail = " …" if n > max_len else ""
        return f"[{head}{tail}]"
    return f"shape={tuple(arr.shape)}"





class StatisticalValidator:
    def __init__(self, config: Configuration):
        """Initialize validator focused on CV execution and checkpointing."""
        self.config = config
        self._validate_config()
        self.current_cv = 1
        self.current_fold = 0
        self.completed_folds = set()
        self._training_completed = False

        # Initialize strategic competition weights and property names
        self.property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.strategic_weights = {
            'Tg': 1.0,
            'FFV': 0.073,
            'Tc': 0.693,
            'Density': 0.834,
            'Rg': 0.832,
        }

        self.config.property_names = self.property_names
        self.config.competition_weights = self.strategic_weights

        self._logger = get_logger(__name__)
        for prop, w in self.strategic_weights.items():
            self._logger.info(f"Strategic weight {prop}: {w}")

        # Setup paths
        self._construct_output_paths()

        # Initialize logger
        LoggerSetup.initialize(
            self.cv_base_dir,
            f"{self.config.task_name}_{self.config.task_type}_cv{self.current_cv}"
        )

        # Initialize data
        self.data_module = MoleculeDataModule(config)
        self.data_module.setup()
        self.train_dataset = self.data_module.train_dataset

        # Load best hyperparameters
        self._load_best_hyperparameters()

        # Initialize results storage
        self.cv_results = []
        self.fold_metrics = []
        self.all_run_metrics = []

        # Initialize additional state variables
        self.current_initialization = 1
        self.seeds = self.config.seed_list
        self.best_metric = float('-inf')
        self.cv_splits = None
        self.fold_splits = None
        self.best_model_state_dict = None

        # Register signal handlers for HPC environment
        signal.signal(signal.SIGTERM, self._handle_termination)
        signal.signal(signal.SIGUSR1, self._handle_termination)
        signal.signal(signal.SIGINT, self._handle_termination)

        # Create checkpoint paths
        self.checkpoint_base_dir = os.path.join(self.cv_base_dir, 'checkpoints')
        self.checkpoint_dir = os.path.join(self.checkpoint_base_dir, f'cv{self.current_cv}')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Create checkpoint lock file
        self.checkpoint_lock = os.path.join(self.cv_base_dir, '.checkpoint.lock')

        # Initialize checkpoint path and load checkpoint
        self.checkpoint_path = os.path.join(self.cv_base_dir, 'checkpoint.json')
        self._load_checkpoint()

        # Define the CSV file path
        self.metrics_csv_path = os.path.join(
            self.cv_base_dir,
            f"{self.config.task_name}_{self.config.task_type}_all_run_metrics.csv"
        )


    def _validate_config(self):
        """Validate configuration parameters"""
        required_params = [
            'task_name',
            'task_type',
            'classification',
            'batch_size',
            'early_stop_patience',
            'max_epochs',
            'statistical_validation'
        ]

        for param in required_params:
            if not hasattr(self.config, param):
                raise ValueError(f"Missing required configuration parameter: {param}")

        if not isinstance(self.config.statistical_validation, dict):
            raise ValueError("statistical_validation must be a dictionary")

        required_stat_params = ['cv_repeats', 'cv_folds']
        for param in required_stat_params:
            if param not in self.config.statistical_validation:
                raise ValueError(f"Missing required statistical validation parameter: {param}")

    def _construct_output_paths(self):
        """Construct paths for saving results and checkpoints."""
        # Base directory for CV results
        self.cv_base_dir = os.path.join(
            self.config.output_dir,
            f"{self.config.task_name}_{self.config.task_type}_cv_results"
        )

        # Create necessary directories
        os.makedirs(self.cv_base_dir, exist_ok=True)

        # Set up paths for current CV
        self._setup_cv_paths(self.current_cv)

    def _setup_cv_paths(self, cv_num: int):
        """Set up paths for a specific CV run."""
        self.cv_run_dir = os.path.join(self.cv_base_dir, f"cv{cv_num}")
        os.makedirs(self.cv_run_dir, exist_ok=True)

        # Checkpoints directory
        self.checkpoints_dir = os.path.join(self.cv_run_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Results paths
        self.results_paths = {
            'intermediate': os.path.join(
                self.cv_run_dir,
                f"{self.config.task_name}_{self.config.task_type}_cv{cv_num}_intermediate.json"
            ),
            'metrics': os.path.join(
                self.cv_run_dir,
                f"{self.config.task_name}_{self.config.task_type}_cv{cv_num}_metrics.csv"
            )
        }




    def _load_best_hyperparameters(self):
        """Load best hyperparameters from the latest results CSV file."""
        # Find the latest results file matching the pattern
        results_dir = self.config.output_dir
        pattern = f"{self.config.task_name}_{self.config.task_type}_*_results_*.csv"
        csv_files = glob.glob(os.path.join(results_dir, pattern))

        if not csv_files:
            raise FileNotFoundError(
                f"No results CSV files found in {results_dir} matching pattern {pattern}"
            )

        # Get the latest file based on the timestamp in filename
        latest_file = max(csv_files, key=os.path.getmtime)
        self._logger.info(f"Loading best hyperparameters from: {latest_file}")

        try:
            # Read CSV and validate columns
            results_df = pd.read_csv(latest_file)
            lr_col = 'learning_rate' if 'learning_rate' in results_df.columns else 'lr'
            required_columns = [
                'rgcn_hidden_feats', 'ffn_hidden_feats', 'ffn_dropout',
                'rgcn_dropout', lr_col, 'weight_decay'
            ]
            missing_columns = [col for col in required_columns if col not in results_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")

            # Find best trial based on metric
            metric_col = 'val_mcc' if self.config.classification else 'val_wmae'
            if metric_col not in results_df.columns:
                raise ValueError(f"Metric column '{metric_col}' not found in CSV")

            # Get best trial (maximize for both MCC and R²)
            best_idx = results_df[metric_col].astype(float).idxmax() if self.config.classification else results_df[metric_col].astype(float).idxmin()

            best_trial = results_df.iloc[best_idx]

            # Parse RGCN hidden features from string to list of integers
            rgcn_hidden_feats = [int(x) for x in str(best_trial['rgcn_hidden_feats']).split('-')]

            # Extract and validate hyperparameters
            self.best_hyperparams = {
                'rgcn_hidden_feats': rgcn_hidden_feats,  # Now a list of integers
                'ffn_hidden_feats': int(best_trial['ffn_hidden_feats']),
                'ffn_dropout': float(best_trial['ffn_dropout']),
                'rgcn_dropout': float(best_trial['rgcn_dropout']),
                'lr': float(best_trial[lr_col]),
                'weight_decay': float(best_trial['weight_decay'])
            }

            # Validate numeric parameters
            for param in ['ffn_dropout', 'rgcn_dropout']:
                if not (0 <= self.best_hyperparams[param] <= 1):
                    raise ValueError(f"Invalid {param} value: {self.best_hyperparams[param]}")

            # Log the loaded parameters
            self._logger.info(f"Best {metric_col}: {best_trial[metric_col]}")
            self._logger.info("Loaded hyperparameters:")
            for k, v in self.best_hyperparams.items():
                self._logger.info(f"  {k}: {v}")

            return self.best_hyperparams

        except Exception as e:
            self._logger.error(f"Error loading hyperparameters from CSV: {e}")
            raise RuntimeError(
                f"Failed to load hyperparameters from {latest_file}. Cannot proceed with validation."
            )




    def _save_checkpoint(self):
        """Save current state to checkpoint with robust error handling."""
        temp_checkpoint_path = self.checkpoint_path + '.tmp'

        try:
            # Prepare checkpoint data
            checkpoint_data = {
                'current_cv': self.current_cv,
                'current_fold': self.current_fold,
                'completed_folds': list(self.completed_folds),
                'cv_results': self.cv_results,
                'fold_metrics': self.fold_metrics,
                'training_completed': self._training_completed,
                'all_run_metrics': self.all_run_metrics,
                'current_initialization': self.current_initialization,
                'seeds': self.seeds,
                'best_metric': self.best_metric,
                'best_model_state_dict': self.best_model_state_dict,
                'property_names': self.config.property_names,
                'competition_weights': self.config.competition_weights
            }

            # Include OOF arrays if they exist (regression only)
            if hasattr(self, '_oof_preds') and self._oof_preds is not None:
                checkpoint_data['oof_preds'] = {k: v.tolist() for k, v in self._oof_preds.items()}
            if hasattr(self, '_oof_labels') and self._oof_labels is not None:
                checkpoint_data['oof_labels'] = {k: v.tolist() for k, v in self._oof_labels.items()}
            if hasattr(self, '_oof_folds') and self._oof_folds is not None:
                checkpoint_data['oof_folds'] = self._oof_folds.tolist()
            if hasattr(self, '_oof_embeddings') and self._oof_embeddings is not None:
                checkpoint_data['oof_embeddings'] = self._oof_embeddings.tolist()

            # First write to temporary file
            with open(temp_checkpoint_path, 'w') as f:
                json.dump(make_serializable(checkpoint_data), f, indent=4)

            # If successful, rename to actual checkpoint file
            if os.path.exists(self.checkpoint_path):
                os.replace(temp_checkpoint_path, self.checkpoint_path)
            else:
                os.rename(temp_checkpoint_path, self.checkpoint_path)

            self._logger.info(f"Checkpoint saved successfully to {self.checkpoint_path}")

        except Exception as e:
            self._logger.error(f"Failed to save checkpoint: {e}")
            if os.path.exists(temp_checkpoint_path):
                try:
                    os.remove(temp_checkpoint_path)
                except Exception as cleanup_error:
                    self._logger.error(f"Failed to clean up temporary checkpoint: {cleanup_error}")


    def _handle_termination(self, signum, frame):
        """Enhanced termination handler for HPC environment."""
        signal_names = {
            signal.SIGTERM: 'SIGTERM (scancel)',
            signal.SIGUSR1: 'SIGUSR1 (time limit)',
            signal.SIGINT: 'SIGINT (keyboard interrupt)'
        }

        signal_name = signal_names.get(signum, f'Unknown signal {signum}')
        self._logger.info(f"\nReceived {signal_name}. Initiating graceful shutdown...")

        try:
            # Create lock file to indicate checkpoint in progress
            with open(self.checkpoint_lock, 'w') as f:
                f.write(str(datetime.now()))

            # Save checkpoint
            self._save_checkpoint()

            # Remove lock file after successful checkpoint
            if os.path.exists(self.checkpoint_lock):
                os.remove(self.checkpoint_lock)

            self._logger.info("Checkpoint saved successfully")

        except Exception as e:
            self._logger.error(f"Error during checkpoint saving: {str(e)}")

        finally:
            # Cleanup
            self.cleanup()
            sys.exit(0)



    def _load_checkpoint(self):
        """Enhanced checkpoint loading for HPC environment."""
        if os.path.exists(self.checkpoint_lock):
            self._logger.warning("Found checkpoint lock file - previous shutdown may have been incomplete")
            time.sleep(5)

        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)

                # Validate checkpoint data
                required_keys = {
                    'current_cv', 'current_fold', 'completed_folds',
                    'cv_results', 'fold_metrics', 'training_completed'
                }

                if not all(key in checkpoint_data for key in required_keys):
                    raise ValueError("Checkpoint file is missing required keys")

                # Restore state
                self.current_cv = checkpoint_data.get('current_cv', 1)
                self.current_fold = checkpoint_data.get('current_fold', 0)
                self.completed_folds = set(checkpoint_data.get('completed_folds', []))
                self.cv_results = checkpoint_data.get('cv_results', [])
                self.fold_metrics = checkpoint_data.get('fold_metrics', [])
                self._training_completed = checkpoint_data.get('training_completed', False)
                self.all_run_metrics = checkpoint_data.get('all_run_metrics', [])
                self.current_initialization = checkpoint_data.get('current_initialization', 1)
                self.seeds = checkpoint_data.get('seeds', self.config.seed_list)
                self.best_metric = checkpoint_data.get('best_metric', float('-inf'))
                self.best_model_state_dict = checkpoint_data.get('best_model_state_dict', None)
                if 'property_names' in checkpoint_data:
                    self.config.property_names = checkpoint_data['property_names']
                if 'competition_weights' in checkpoint_data:
                    self.config.competition_weights = checkpoint_data['competition_weights']

                # Restore OOF arrays if they exist (regression only)
                if 'oof_preds' in checkpoint_data:
                    self._oof_preds = {k: np.array(v, dtype=np.float32) for k, v in checkpoint_data['oof_preds'].items()}
                if 'oof_labels' in checkpoint_data:
                    self._oof_labels = {k: np.array(v, dtype=np.float32) for k, v in checkpoint_data['oof_labels'].items()}
                if 'oof_folds' in checkpoint_data:
                    self._oof_folds = np.array(checkpoint_data['oof_folds'], dtype=np.int32)
                if 'oof_embeddings' in checkpoint_data:
                    self._oof_embeddings = np.array(checkpoint_data['oof_embeddings'], dtype=np.float32)

                # Update checkpoint directory for current CV
                self.checkpoint_dir = os.path.join(self.checkpoint_base_dir, f'cv{self.current_cv}')
                os.makedirs(self.checkpoint_dir, exist_ok=True)

                # Backup existing checkpoint
                backup_path = f"{self.checkpoint_path}.{int(time.time())}.bak"
                shutil.copy2(self.checkpoint_path, backup_path)

                self._setup_cv_paths(self.current_cv)
                self._logger.info(
                    f"Resumed from checkpoint: CV{self.current_cv}, Fold {self.current_fold}"
                    f"\nCompleted folds: {sorted(list(self.completed_folds))}"
                )

            except Exception as e:
                self._logger.error(f"Failed to load checkpoint: {e}")
                self._backup_corrupted_checkpoint()
                self._initialize_fresh_state()
        else:
            self._initialize_fresh_state()

    def _backup_corrupted_checkpoint(self):
        """Backup corrupted checkpoint file."""
        if os.path.exists(self.checkpoint_path):
            backup_path = f"{self.checkpoint_path}.corrupted.{int(time.time())}"
            try:
                shutil.move(self.checkpoint_path, backup_path)
                self._logger.info(f"Backed up corrupted checkpoint to {backup_path}")
            except Exception as e:
                self._logger.error(f"Failed to backup corrupted checkpoint: {e}")

    def _initialize_fresh_state(self):
        """Initialize fresh state when no checkpoint exists or loading fails."""
        self._logger.info("Initializing fresh state")
        self.current_cv = 1
        self.current_fold = 0
        self.completed_folds = set()
        self.cv_results = []
        self.fold_metrics = []
        self._training_completed = False
        # Ensure config still contains weights and property names
        if not hasattr(self.config, 'competition_weights'):
            self.config.competition_weights = {
                'Tg': 1.0,
                'FFV': 0.073,
                'Tc': 0.693,
                'Density': 0.834,
                'Rg': 0.832,
            }
        if not hasattr(self.config, 'property_names'):
            self.config.property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']





    def perform_cross_validation(self):
        """Execute complete cross-validation procedure."""
        try:
            total_cvs = self.config.statistical_validation['cv_repeats']
            total_folds = self.config.statistical_validation['cv_folds']

            # Use only 1 training per fold with best hyperparameters (not multiple initializations)
            total_runs = total_cvs * total_folds

            self._logger.info(f"\nStarting complete validation:")
            self._logger.info(f"Number of CVs: {total_cvs}")
            self._logger.info(f"Number of folds per CV: {total_folds}")
            self._logger.info(f"Using best hyperparameters from optimization")
            self._logger.info(f"Total number of model trainings: {total_runs}")
            self._logger.info(f"Deterministic seed: 42")

            # Iterate through CVs
            for cv_num in range(self.current_cv, total_cvs + 1):
                try:
                    self.current_cv = cv_num
                    self._setup_cv_paths(cv_num)
                    self._logger.info(f"\n{'='*50}")
                    self._logger.info(f"Starting CV {cv_num} of {total_cvs}")
                    self._logger.info(f"{'='*50}")
                    # Delegate the full CV iteration (handles folds, OOF, embeddings export)
                    self._perform_cv_iterations()
                    # After completing all folds in current CV, record metrics snapshot
                    self.cv_results.append({
                        'cv': cv_num,
                        'metrics': self.fold_metrics.copy()
                    })
                    # Reset for next CV
                    self.fold_metrics = []
                    self.current_fold = 0
                    self.completed_folds = set()
                    self._save_checkpoint()
                except Exception as e:
                    self._logger.error(f"Error in CV {cv_num}: {str(e)}")
                    self._save_checkpoint()
                    raise

            self._training_completed = True
            self._save_checkpoint()
            self._logger.info("\nAll cross-validation runs completed successfully")

            # Save final results
            final_results = {
                'config': {
                    'task_name': self.config.task_name,
                    'task_type': self.config.task_type,
                    'classification': self.config.classification,
                    'hyperparameters': self.best_hyperparams,
                    'cv_structure': {
                        'cv_repeats': total_cvs,
                        'cv_folds': total_folds,
                        'seed': 42
                    }
                },
                'cv_results': self.cv_results,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            final_results_path = os.path.join(
                self.cv_base_dir,
                f"{self.config.task_name}_{self.config.task_type}_final_results.json"
            )
            with open(final_results_path, 'w') as f:
                json.dump(make_serializable(final_results), f, indent=4)

        except Exception as e:
            self._logger.error(f"Error during cross-validation: {str(e)}")
            self._save_checkpoint()
            raise

    def _perform_cv_iterations(self):
        """Perform k-fold CV for current CV iteration."""
        n_splits = self.config.statistical_validation['cv_folds']

        try:
            # Load and validate data from processed location (under graphs_4edge)
            meta_path = self.config.get_processed_file_path('meta', 'primary')
            if not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"Processed meta not found at: {meta_path}. "
                    f"Ensure build/prep steps created the processed files under output_dir."
                )
            self._logger.info(f"Loading processed meta for CV from: {meta_path}")
            meta_data = pd.read_csv(meta_path)

            training_data = meta_data[meta_data['group'] == 'training']
            labels = training_data[self.config.labels_name].values
            indices = np.arange(len(self.train_dataset))
            start_time = datetime.now()

            # Create splitter for current CV
            if self.config.classification:
                splitter = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=42 * self.current_cv  # Unique seed for each CV
                )
            else:
                splitter = KFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=42 * self.current_cv
                )

            # Get all folds for current CV
            if self.config.classification:
                fold_splits = list(splitter.split(indices, labels))
            else:
                fold_splits = list(splitter.split(indices))

            self._logger.info(f"\nCV {self.current_cv}: Starting {n_splits} folds")
            self._logger.info(f"Training once per fold with best hyperparameters (seed=42)")

            # Prepare OOF aggregators for regression
            meta_df = meta_data
            train_mask = (meta_df['group'] == 'training').values
            train_meta_indices = np.arange(len(meta_df))[train_mask]
            n_train = len(self.train_dataset)
            if not self.config.classification:
                self._oof_preds = {p: np.full(n_train, np.nan, dtype=np.float32) for p in self.config.property_names}
                self._oof_labels = {p: np.full(n_train, np.nan, dtype=np.float32) for p in self.config.property_names}
                self._oof_folds = np.full(n_train, -1, dtype=np.int32)  # Track which fold each sample came from
                self._oof_embeddings = None

            for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
                if fold_idx < self.current_fold:
                    continue

                self.current_fold = fold_idx

                # Create data subsets
                train_subset = Subset(self.train_dataset, train_idx)
                val_subset = Subset(self.train_dataset, val_idx)

                self._logger.info(f"\nCV {self.current_cv}, Fold {fold_idx + 1}/{n_splits}")
                self._logger.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")

                # Train model with multiple initializations
                metrics = self._train_fold_model(train_subset, val_subset, fold_idx)

                # Update progress and save
                self.fold_metrics.append(metrics)
                self.completed_folds.add(fold_idx)
                self._save_checkpoint()
                self._save_intermediate_results()
                self._log_fold_progress(fold_idx + 1, n_splits, start_time)
                self._cleanup_fold_resources()

                # Fill OOF arrays for this fold (regression only)
                if not self.config.classification:
                    for prop in self.config.property_names:
                        fold_preds = np.asarray(metrics['predictions'][prop], dtype=np.float32)
                        fold_labels = np.asarray(metrics['labels'][prop], dtype=np.float32)
                        self._oof_preds[prop][val_idx] = fold_preds
                        self._oof_labels[prop][val_idx] = fold_labels

                    # Track fold information for each validation sample
                    self._oof_folds[val_idx] = fold_idx + 1  # Use 1-based fold numbering

                    # --- OOF embeddings aggregation ---
                    emb = metrics.get("embeddings", None)
                    if emb is not None:
                        val_idx_array = np.asarray(val_idx, dtype=int)
                        if self._oof_embeddings is None:
                            emb_dim = emb.shape[1]
                            self._oof_embeddings = np.full((n_train, emb_dim), np.nan, dtype=np.float32)
                            self._logger.info(f"[cv {self.current_cv}] initialized OOF embeddings: shape={self._oof_embeddings.shape}")
                        self._oof_embeddings[val_idx_array, :] = emb
                        self._logger.info(f"[cv {self.current_cv} fold {fold_idx}] wrote embeddings rows: {len(val_idx_array)}")

            # Reset for next CV
            if len(self.completed_folds) == n_splits:
                self.current_fold = 0
                self.completed_folds = set()
                self._logger.info(f"\nCompleted all {n_splits} folds for CV {self.current_cv}")

                # Exports after all folds (regression only)
                try:
                    # Label coverage counts over full dataset
                    label_counts = {}
                    for prop in self.config.property_names:
                        total = len(meta_df)
                        non_na = int(meta_df[prop].notna().sum()) if prop in meta_df.columns else 0
                        label_counts[prop] = {
                            'count': non_na,
                            'coverage': float(non_na) / float(total) if total else 0.0,
                            'total': total,
                        }
                    with open(os.path.join(self.cv_base_dir, 'label_counts.json'), 'w') as f:
                        json.dump(label_counts, f, indent=2)

                    if not self.config.classification:
                        # OOF predictions parquet (train set rows only)
                        # Check if fold tracking is available
                        has_fold_tracking = hasattr(self, '_oof_folds') and self._oof_folds is not None
                        if not has_fold_tracking:
                            self._logger.warning("No fold tracking available - fold column will be NaN")

                        rows = []
                        for i in range(n_train):
                            meta_i = int(train_meta_indices[i])
                            row = {
                                'meta_index': meta_i,
                                self.config.compound_id_name: meta_df.loc[meta_i, self.config.compound_id_name]
                            }

                            # Add fold information if available
                            if has_fold_tracking:
                                row['fold'] = int(self._oof_folds[i]) if self._oof_folds[i] != -1 else np.nan
                            else:
                                row['fold'] = np.nan
                            for prop in self.config.property_names:
                                row[f'{prop}_oof'] = float(self._oof_preds[prop][i]) if np.isfinite(self._oof_preds[prop][i]) else np.nan
                                row[f'{prop}_label'] = float(self._oof_labels[prop][i]) if np.isfinite(self._oof_labels[prop][i]) else np.nan
                            rows.append(row)
                        oof_df = pd.DataFrame(rows)
                        oof_path = os.path.join(self.cv_run_dir, 'oof_predictions.parquet')
                        try:
                            oof_df.to_parquet(oof_path, index=False)
                        except Exception:
                            # Fallback to CSV if parquet not available
                            oof_path = os.path.join(self.cv_run_dir, 'oof_predictions.csv')
                            oof_df.to_csv(oof_path, index=False)
                        self._logger.info(f"Saved OOF predictions to {oof_path}")

                        # Embeddings export
                        if self._oof_embeddings is not None:
                            emb_path = os.path.join(self.cv_run_dir, 'embeddings.npy')
                            np.save(emb_path, self._oof_embeddings)
                            self._logger.info(f"Saved embeddings to {emb_path}")
                        else:
                            self._logger.warning(f"No embeddings collected — embeddings.npy will not be written for {self.cv_run_dir}")
                except Exception as e:
                    self._logger.warning(f"Post-fold export failed: {e}")

        except Exception as e:
            self._logger.error(f"Error in CV iterations: {str(e)}")
            raise



    def _train_fold_model(self, train_subset: Subset, val_subset: Subset, fold_idx: int) -> Dict:
        """Train model for a specific fold using best hyperparameters."""

        # Calculate appropriate log_every_n_steps
        num_batches = len(train_subset) // self.config.batch_size
        log_every_n_steps = max(1, num_batches // 10)

        # Use fixed seed for reproducibility
        seed = 42

        try:
            # Create a temporary DataModule for this fold
            class FoldDataModule(pl.LightningDataModule):
                def __init__(self, train_subset, val_subset, batch_size, num_workers, collate_fn):
                    super().__init__()
                    self.train_subset = train_subset
                    self.val_subset = val_subset
                    self.batch_size = batch_size
                    self.num_workers = num_workers
                    self.collate_fn = collate_fn

                def train_dataloader(self):
                    return DataLoader(
                        self.train_subset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.num_workers,
                        pin_memory=True,
                        collate_fn=self.collate_fn,
                        persistent_workers=True if self.num_workers > 0 else False
                    )

                def val_dataloader(self):
                    return DataLoader(
                        self.val_subset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers,
                        pin_memory=True,
                        collate_fn=self.collate_fn,
                        persistent_workers=True if self.num_workers > 0 else False
                    )

            # Create DataModule instance for this fold
            fold_datamodule = FoldDataModule(
                train_subset=train_subset,
                val_subset=val_subset,
                batch_size=self.config.batch_size,
                num_workers=self.data_module.num_workers,
                collate_fn=collate_fn
            )

            # Set up logging directory
            log_dir = os.path.join(self.cv_run_dir, 'lightning_logs')
            os.makedirs(log_dir, exist_ok=True)

            # Set seed for reproducibility
            pl.seed_everything(seed, workers=True)
            self._logger.info(f"\nTraining fold {fold_idx + 1} with seed {seed}")

            # Create model with all required parameters
            try:
                model = BaseGNN(
                    config=self.config,
                    rgcn_hidden_feats=self.best_hyperparams['rgcn_hidden_feats'],
                    ffn_hidden_feats=self.best_hyperparams['ffn_hidden_feats'],
                    ffn_dropout=self.best_hyperparams['ffn_dropout'],
                    rgcn_dropout=self.best_hyperparams['rgcn_dropout'],
                    classification=self.config.classification,
                    num_classes=2 if self.config.classification else None
                )
            except Exception as e:
                self._logger.error(f"Error creating model: {str(e)}")
                self._logger.error("Current hyperparameters:")
                for k, v in self.best_hyperparams.items():
                    self._logger.error(f"  {k}: {v}")
                raise

            # Configure callbacks
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.checkpoints_dir,
                filename=f"temp_cv{self.current_cv}_fold{fold_idx}",
                save_top_k=1,
                monitor='val_mcc' if self.config.classification else 'val_wmae',
                mode='max',  # Maximize for both MCC and R²
                save_last=False
            )

            early_stop = EarlyStopping(
                monitor='val_mcc' if self.config.classification else 'val_wmae',
                mode='max',  # Maximize for both MCC and R²
                patience=self.config.early_stop_patience,
                verbose=True
            )

            # Create a custom logger name for this fold
            version = f"cv{self.current_cv}_fold{fold_idx}"
            pl_logger = pl.loggers.TensorBoardLogger(
                save_dir=log_dir,
                name=f"{self.config.task_name}_{self.config.task_type}",
                version=version,
                default_hp_metric=False
            )

            # Override callbacks to minimize wMAE for regression
            if not self.config.classification:
                try:
                    checkpoint_callback.monitor = 'val_wmae'
                    checkpoint_callback.mode = 'min'
                    early_stop.monitor = 'val_wmae'
                    early_stop.mode = 'min'
                    self._logger.info("Using val_wmae with mode=min for regression")
                except Exception as _e:
                    self._logger.warning(f"Could not set regression monitors to wMAE: {_e}")

            trainer = pl.Trainer(
                max_epochs=self.config.max_epochs,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                callbacks=[checkpoint_callback, early_stop],
                enable_progress_bar=True,
                deterministic=False,
                enable_checkpointing=True,
                logger=pl_logger,
                num_sanity_val_steps=0,
                log_every_n_steps=log_every_n_steps,
            )

            # Train model using the DataModule
            trainer.fit(model, datamodule=fold_datamodule)

            # Evaluation code
            metrics = self._evaluate_fold(model, val_subset, fold_idx)

            # Record run metrics matching CSV structure based on task type
            if self.config.classification:
                run_metrics = {
                    'CV': self.current_cv,
                    'Fold': fold_idx + 1,
                    'Init': 1,
                    'Seed': seed,
                    'AUC': metrics['auc'],
                    'MCC': metrics['mcc'],
                    'Acc': metrics['accuracy'],
                    'Kappa': metrics['kappa'],
                    'Prec': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1': metrics['f1'],
                    'Spec': metrics['specificity'],
                    'Saved': 'Yes'
                }
            else:
                run_metrics = {
                    'CV': self.current_cv,
                    'Fold': fold_idx + 1,
                    'Init': 1,
                    'Seed': seed,
                    'WMAE': metrics['wmae'],
                    'Tg_MAE': metrics['mae_Tg'],
                    'FFV_MAE': metrics['mae_FFV'],
                    'Tc_MAE': metrics['mae_Tc'],
                    'Density_MAE': metrics['mae_Density'],
                    'Rg_MAE': metrics['mae_Rg'],
                    'Saved': 'Yes'
                }

            self.all_run_metrics.append(run_metrics)

            # Save the model checkpoint
            model_state_dict = copy.deepcopy(model.state_dict())

            # Cleanup
            del model
            if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
                try:
                    os.remove(checkpoint_callback.best_model_path)
                except Exception as e:
                    self._logger.warning(f"Failed to remove temporary checkpoint: {e}")

            torch.cuda.empty_cache()

            # Save the model checkpoint
            if model_state_dict is not None:
                # Convert to 1-based fold number for filenames
                fold_num = fold_idx + 1  # Convert from 0-based to 1-based for filename

                # Clean up existing checkpoints for this fold
                existing_checkpoints = glob.glob(os.path.join(
                    self.checkpoints_dir,
                    f"{self.config.task_name}_{self.config.task_type}_cv{self.current_cv}_fold{fold_num}_*.ckpt"
                ))
                for checkpoint in existing_checkpoints:
                    try:
                        os.remove(checkpoint)
                    except Exception as e:
                        self._logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")

                # Save the model
                self._save_model_checkpoint(model_state_dict, fold_idx, init_idx=0)

            # Write the run metrics to CSV
            self._write_run_metrics_csv()

            return metrics

        except Exception as e:
            self._logger.error(f"Error in train fold model: {str(e)}")
            self._logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

    def _evaluate_fold(self, model: BaseGNN, val_subset: Subset, fold_idx: int) -> Dict:
        """Evaluate model on validation fold."""
        # Ensure model is in eval mode
        model.eval()

        # Create DataLoader with proper settings
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.data_module.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.data_module.num_workers > 0 else False
        )

        all_preds = {prop: [] for prop in self.config.property_names} if not self.config.classification else []
        all_labels = {prop: [] for prop in self.config.property_names} if not self.config.classification else []
        all_embeds = []
        collect_embeddings = not self.config.classification

        # Get model device for device safety
        device = next(model.parameters()).device

        try:
            with torch.no_grad():
                for batch in val_loader:
                    graphs, labels = batch

                    # Move batch tensors to model device
                    graphs = graphs.to(device)
                    labels = labels.to(device)

                    # Get forward output - now returns (preds, penult)
                    out = model(graphs)
                    if isinstance(out, tuple) and len(out) == 2:
                        outputs, penult = out
                    else:
                        outputs = out
                        penult = model.get_embeddings()

                    if self.config.classification:
                        preds = torch.sigmoid(outputs)
                        all_preds.extend(preds.cpu().detach().numpy().flatten())
                        all_labels.extend(labels.cpu().detach().numpy())
                    else:
                        for idx, prop in enumerate(self.config.property_names):
                            prop_pred = outputs[prop].cpu().detach().numpy()
                            prop_label = labels[:, idx].cpu().detach().numpy()
                            all_preds[prop].extend(prop_pred)
                            all_labels[prop].extend(prop_label)

                        # Collect embeddings from forward pass
                        if collect_embeddings and penult is not None:
                            all_embeds.append(penult.detach().cpu().numpy())

            if self.config.classification:
                predictions = np.array(all_preds)
                labels = np.array(all_labels)

                metrics = {
                    'fold': fold_idx + 1,
                    'cv': self.current_cv,
                    'predictions': predictions.tolist(),
                    'labels': labels.tolist()
                }

                binary_preds = predictions > 0.5
                metrics.update({
                    'auc': float(roc_auc_score(labels, predictions)),
                    'accuracy': float(accuracy_score(labels, binary_preds)),
                    'precision': float(precision_score(labels, binary_preds)),
                    'recall': float(recall_score(labels, binary_preds)),
                    'f1': float(f1_score(labels, binary_preds)),
                    'specificity': float(recall_score(labels, binary_preds, pos_label=0)),
                    'mcc': float(matthews_corrcoef(labels, binary_preds)),
                    'kappa': float(cohen_kappa_score(labels, binary_preds))
                })
            else:
                metrics = {
                    'fold': fold_idx + 1,
                    'cv': self.current_cv,
                    'predictions': {prop: vals for prop, vals in all_preds.items()},
                    'labels': {prop: vals for prop, vals in all_labels.items()}
                }
                # Attach embeddings if collected
                if collect_embeddings and len(all_embeds) > 0:
                    try:
                        metrics['embeddings'] = np.vstack(all_embeds)
                        self._logger.info(f"[fold {fold_idx}] collected embeddings shape: {metrics['embeddings'].shape}")
                    except Exception:
                        pass

                maes = {}
                total_loss = 0.0
                total_weight = 0.0
                for prop in self.config.property_names:
                    preds_arr = np.array(all_preds[prop])
                    labels_arr = np.array(all_labels[prop])
                    mask = np.isfinite(labels_arr)
                    if mask.any():
                        mae = mean_absolute_error(labels_arr[mask], preds_arr[mask])
                        weight = self.config.competition_weights.get(prop, 1.0)
                        total_loss += weight * mae
                        total_weight += weight
                        maes[prop] = mae
                    else:
                        maes[prop] = 0.0

                wmae = total_loss / total_weight if total_weight > 0 else 0.0
                metrics['wmae'] = wmae
                for prop, mae in maes.items():
                    metrics[f'mae_{prop}'] = mae

            # Log metrics without predictions and labels for readability
            log_metrics = {k: v for k, v in metrics.items() if k not in ['predictions', 'labels', 'embeddings']}
            self._logger.info(f"Fold {fold_idx + 1} Metrics:")
            for metric_name, value in log_metrics.items():
                if metric_name not in ['fold', 'cv']:
                    formatted_value = _fmt_metric_value(value)
                    self._logger.info(f"  {metric_name}: {formatted_value}")

            return metrics

        except Exception as e:
            self._logger.error(f"Error during evaluation: {str(e)}")
            self._logger.error(f"Traceback:\n{traceback.format_exc()}")

            # Log debugging information
            try:
                self._logger.error(f"Model device: {next(model.parameters()).device}")
                if 'graphs' in locals() and isinstance(graphs, Batch):
                    self._logger.error(f"Graphs device: {graphs.x.device}")
                    self._logger.error(f"Available keys: {list(graphs.keys())}")
                if 'labels' in locals():
                    self._logger.error(f"Labels device: {labels.device}")
                    self._logger.error(f"Batch size: {len(labels)}")
            except Exception as debug_error:
                self._logger.error(f"Error during debug logging: {str(debug_error)}")

            raise





    def _save_model_checkpoint(self, state_dict: Dict[str, Any], fold_idx: int, init_idx: int):
        """Save model checkpoint using 1-based fold indexing in filename."""
        try:
            # Convert to 1-based fold number for checkpoint filename
            fold_num = fold_idx + 1  # Convert from 0-based to 1-based for filename

            checkpoint_path = os.path.join(
                self.checkpoints_dir,
                f"{self.config.task_name}_{self.config.task_type}_cv{self.current_cv}_fold{fold_num}_best.ckpt"
            )

            # Create temporary checkpoint path
            temp_checkpoint_path = checkpoint_path + '.tmp'

            # Save to temporary file first
            checkpoint_data = {
                'state_dict': state_dict,
                'hyperparameters': self.best_hyperparams,
                'config': {
                    'task_name': self.config.task_name,
                    'task_type': self.config.task_type,
                    'cv': self.current_cv,
                    'fold': fold_num,  # Store 1-based fold number
                    'init': init_idx,
                    'classification': self.config.classification,
                    'property_names': self.config.property_names,
                    'competition_weights': self.config.competition_weights
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            torch.save(checkpoint_data, temp_checkpoint_path)

            # If successful, rename to actual checkpoint file
            if os.path.exists(checkpoint_path):
                os.replace(temp_checkpoint_path, checkpoint_path)
            else:
                os.rename(temp_checkpoint_path, checkpoint_path)

            self._logger.info(f"Saved best model checkpoint: {checkpoint_path}")

            # Verify checkpoint
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file was not saved: {checkpoint_path}")

        except Exception as e:
            self._logger.error(f"Error in checkpoint saving: {e}")
            # Clean up temporary file if it exists
            if os.path.exists(temp_checkpoint_path):
                try:
                    os.remove(temp_checkpoint_path)
                except Exception as cleanup_error:
                    self._logger.error(f"Failed to clean up temporary checkpoint: {cleanup_error}")
            raise



    def _save_intermediate_results(self):
        """Save intermediate results during training."""
        results = {
            'config': {
                'task_name': self.config.task_name,
                'task_type': self.config.task_type,
                'cv': self.current_cv,
                'hyperparameters': self.best_hyperparams
            },
            'completed_folds': list(self.completed_folds),
            'fold_metrics': self.fold_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save JSON results
        with open(self.results_paths['intermediate'], 'w') as f:
            json.dump(make_serializable(results), f, indent=4)

        # Save metrics CSV without predictions and labels
        metrics_df = pd.DataFrame(self.fold_metrics)
        if not metrics_df.empty:
            metrics_df = metrics_df.drop(['predictions', 'labels'], axis=1, errors='ignore')
        metrics_df.to_csv(self.results_paths['metrics'], index=False)

        self._logger.info(f"Saved intermediate results to {self.cv_run_dir}")






    def _write_run_metrics_csv(self):
        """Write all run metrics to a CSV file with proper error handling."""
        if self.config.classification:
            headers = ['CV', 'Fold', 'Init', 'Seed', 'MCC', 'AUC', 'Acc', 'Kappa', 'Prec',
                      'Recall', 'F1', 'Spec', 'Saved', 'FinalSaved']
        else:
            headers = ['CV', 'Fold', 'Init', 'Seed', 'WMAE', 'Tg_MAE', 'FFV_MAE', 'Tc_MAE',
                       'Density_MAE', 'Rg_MAE', 'Saved', 'FinalSaved']

        try:
            # Create temporary file path
            temp_csv_path = self.metrics_csv_path + '.tmp'

            # Write to temporary file
            # Since we only have one initialization per fold, all runs are final saved
            with open(temp_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()

                for run in self.all_run_metrics:
                    run_copy = run.copy()
                    # Mark as FinalSaved (always 'Yes' since we only train once per fold)
                    run_copy['FinalSaved'] = 'Yes'
                    writer.writerow(run_copy)

            # If successful, rename to actual file
            if os.path.exists(self.metrics_csv_path):
                os.replace(temp_csv_path, self.metrics_csv_path)
            else:
                os.rename(temp_csv_path, self.metrics_csv_path)

            self._logger.info(f"All run metrics saved to CSV at {self.metrics_csv_path}")

        except Exception as e:
            self._logger.error(f"Failed to write run metrics to CSV: {e}")
            if os.path.exists(temp_csv_path):
                try:
                    os.remove(temp_csv_path)
                except Exception as cleanup_error:
                    self._logger.error(f"Failed to clean up temporary CSV file: {cleanup_error}")
            raise

    def _log_fold_progress(self, current_fold: int, total_folds: int, start_time: datetime):
        """Log progress of current fold."""
        elapsed_time = datetime.now() - start_time
        avg_time_per_fold = elapsed_time / current_fold if current_fold > 0 else elapsed_time
        remaining_folds = total_folds - current_fold
        estimated_remaining = avg_time_per_fold * remaining_folds

        self._logger.info(f"\nProgress: Fold {current_fold}/{total_folds}")
        self._logger.info(f"Time per fold: {avg_time_per_fold.total_seconds()/60:.1f} minutes")
        self._logger.info(f"Estimated remaining: {estimated_remaining.total_seconds()/60:.1f} minutes")


    def signal_handler(signum, frame):
        self._logger.info(f"Received signal {signum}. Saving checkpoint and exiting.")
        self._save_checkpoint()
        sys.exit(0)


    def cleanup(self):
        """Clean up resources without using logger."""
        try:
            if hasattr(self, 'data_module'):
                self.data_module.cleanup()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


    def _cleanup_fold_resources(self):
        """Clean up resources after fold completion"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Log memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            self._logger.info(f"GPU Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

        ram = psutil.virtual_memory()
        self._logger.info(f"RAM Usage: {ram.percent}%")





    def __del__(self):
        """Avoid using logger during object deletion."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass  # Suppress all exceptions during interpreter shutdown







if __name__ == "__main__":
    # Initialize configuration
    config = Configuration()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize logger
    LoggerSetup.initialize(config.output_dir, config.task_name)
    logger = get_logger(__name__)

    try:
        validator = StatisticalValidator(config)
        validator.perform_cross_validation()
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
    finally:
        validator.cleanup()
