#data_module.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch_geometric.data import Data, Batch
import os
import pytorch_lightning as pl
from logger import get_logger
from config import Configuration
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import psutil
from pathlib import Path


def _is_pin_memory_safe() -> bool:
    """Return True if pin_memory is supported in the current environment."""
    if os.environ.get("DISABLE_PIN_MEMORY", "0") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        torch.empty(1).pin_memory()
        return True
    except Exception:
        return False

# Initialize logger
logger = get_logger(__name__)
logger.info("Starting data_module process...")


def collate_fn(batch):
    """Optimized collate function for PyG data"""
    graphs, labels = zip(*batch)
    
    # Pre-allocate memory for better efficiency
    batched_graphs = Batch.from_data_list(
        list(graphs), 
        follow_batch=['x', 'edge_index'],
        exclude_keys=['batch']  # Exclude unnecessary attributes
    )
    
    labels = torch.stack(labels)
    
    # Do not move data to GPU here
    
    return batched_graphs, labels

class MoleculeDataset(Dataset):
    """Memory-efficient dataset with smart caching"""
    def __init__(self, config: Configuration, data_path: str, meta_path: str):
        self.config = config
        self._validate_paths(data_path, meta_path)

        # Load metadata (small memory footprint)
        self.meta = pd.read_csv(meta_path)
        logger.info(f"Loaded metadata from: {meta_path}")
        
        # Initialize data storage
        self._data = None
        self._labels = None
        self.pin_memory = False
        
        # Smart loading based on memory availability
        self._initialize_data_storage(data_path)

    def _validate_paths(self, data_path: str, meta_path: str):
        """Validate file paths exist"""
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        self.data_path = data_path

    def _prepare_labels(self, labels):
        """Prepare labels with correct dtype"""
        if isinstance(labels, torch.Tensor):
            labels_tensor = labels.clone().detach()
        else:
            labels_tensor = torch.tensor(labels)
            
        return labels_tensor.to(
            dtype=torch.long if self.config.classification else torch.float32
        )

    def __len__(self):
        return len(self.meta)

    def _initialize_data_storage(self, data_path: str):
        """Initialize data storage with optimized memory usage"""
        file_size = os.path.getsize(data_path)
        available_memory = psutil.virtual_memory().available
        estimated_memory = file_size * 1.2
        
        logger.info(f"Dataset size: {file_size / 1e9:.2f} GB")
        logger.info(f"Available memory: {available_memory / 1e9:.2f} GB")
        
        if estimated_memory < available_memory * 0.85:
            try:
                loaded_data = torch.load(
                    data_path, 
                    map_location='cpu',
                    mmap=True  # Memory-mapped file loading
                )
                self._data = loaded_data[0]

                # Pin memory only if supported
                if _is_pin_memory_safe():
                    try:
                        self._data = [g.contiguous().pin_memory() for g in self._data]
                        self.pin_memory = True
                    except Exception as e:
                        logger.warning(f"Pin memory failed: {e}. Continuing without pinning.")
                        self._data = [g.contiguous() for g in self._data]
                        self.pin_memory = False
                else:
                    self.pin_memory = False
                    self._data = [g.contiguous() for g in self._data]
                
                self._labels = self._prepare_labels(loaded_data[1]['labels'])
                logger.info("Dataset loaded into memory successfully")
                
            except Exception as e:
                logger.error(f"Error loading dataset: {str(e)}")
                raise
        else:
            logger.info("Using deferred loading due to memory constraints")
            self._load_on_demand = True


    def __getitem__(self, idx):
        try:
            # Optimized data loading
            if self._data is None and hasattr(self, '_load_on_demand'):
                # Load only the required chunk
                loaded_data = torch.load(
                    self.data_path, 
                    map_location='cpu',
                    mmap=True
                )
                self._data = loaded_data[0]
                self._labels = self._prepare_labels(loaded_data[1]['labels'])
            
            graph = self._data[idx]
            label = self._labels[idx]
            
            # Optimize batch attribute creation
            if not hasattr(graph, 'batch'):
                graph.batch = torch.zeros(
                    graph.x.size(0), 
                    dtype=torch.long,
                    device=graph.x.device
                )
            
            return graph, label
            
        except Exception as e:
            logger.error(f"Error accessing dataset at idx {idx}: {str(e)}")
            raise

class MoleculeDataModule(pl.LightningDataModule):
    """Optimized Data Module for Molecule Datasets"""
    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config
        self.batch_size = self.config.batch_size  # Start with config batch size
        self.num_workers = self._determine_optimal_workers()
        
        # Initialize dataloader kwargs with initial batch size
        self.pin_memory = False  # will be updated after loading

        self.dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': False,  # Disable by default
            'persistent_workers': self.num_workers > 0,
            'prefetch_factor': 2 if self.num_workers > 0 else None,
            'collate_fn': collate_fn
        }
        
        # Initialize dataset attributes
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.is_cleaned_up = False
        
        logger.info(f"Initializing DataModule for task: {config.task_name}")
        logger.info(f"Task type: {config.task_type}")
        logger.info(f"Initial batch size: {self.batch_size}")
        logger.info(f"Number of workers: {self.num_workers}")

    def _determine_optimal_workers(self) -> int:
        """Optimized worker configuration"""
        cpu_count = os.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if torch.cuda.is_available():
            # Increased workers for GPU training
            optimal_workers = min(8, cpu_count // 2)
            logger.info(f"Using {optimal_workers} workers for GPU training")
        elif memory_gb < 16:
            optimal_workers = min(2, cpu_count // 4)
            logger.info("Limited workers due to low memory")
        else:
            optimal_workers = min(cpu_count - 1, 12)  # Increased max workers
            logger.info(f"Using {optimal_workers} workers for CPU training")
            
        return optimal_workers
        
    
    def update_dataloader_kwargs(self, new_kwargs: Dict[str, Any]) -> None:
        """
        Update dataloader keyword arguments with new values.
        
        Args:
            new_kwargs: Dictionary containing new dataloader arguments
        """
        # Validate the new kwargs
        valid_keys = {
            'batch_size', 'num_workers', 'pin_memory', 
            'persistent_workers', 'prefetch_factor'
        }
        
        invalid_keys = set(new_kwargs.keys()) - valid_keys
        if invalid_keys:
            logger.warning(f"Ignoring invalid dataloader kwargs: {invalid_keys}")
        
        # Update only valid kwargs
        for key, value in new_kwargs.items():
            if key in valid_keys:
                if key == 'num_workers':
                    value = min(value, self._determine_optimal_workers())
                self.dataloader_kwargs[key] = value
                    
        logger.debug(f"Updated dataloader kwargs: {self.dataloader_kwargs}")


    def setup(self, stage: Optional[str] = None):
        """Optimized dataset setup with memory efficiency"""
        if self.train_dataset is not None:
            return
                
        data_path = self.config.get_processed_file_path('graphs', 'primary')
        meta_path = self.config.get_processed_file_path('meta', 'primary')
        
        # Create dataset with optimized settings
        self.dataset = MoleculeDataset(self.config, data_path, meta_path)

        # Update pin_memory setting based on dataset initialization
        if getattr(self.dataset, 'pin_memory', False):
            self.dataloader_kwargs['pin_memory'] = True
        
        # Use numpy for faster indexing and memory efficiency
        meta_df = self.dataset.meta
        train_mask = (meta_df['group'] == 'training').values
        val_mask = (meta_df['group'] == 'valid').values
        test_mask = (meta_df['group'] == 'test').values
        
        # Optimize indexing
        train_indices = np.arange(len(meta_df))[train_mask]
        val_indices = np.arange(len(meta_df))[val_mask]
        test_indices = np.arange(len(meta_df))[test_mask]
        
        # Create optimized subsets
        from torch.utils.data import Subset
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)
        
        logger.info(f"Dataset splits - Train: {len(train_indices)}, "
                   f"Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    def train_dataloader(self):
        """Optimized train dataloader with better memory handling"""
        train_kwargs = {
            **self.dataloader_kwargs,
            'shuffle': True,
            'pin_memory': self.dataloader_kwargs.get('pin_memory', False),
            'drop_last': True,  # Optimize batch processing
            'persistent_workers': True if self.num_workers > 0 else False,
        }
        
        return DataLoader(self.train_dataset, **train_kwargs)
    
    def val_dataloader(self):
        """Optimized validation dataloader"""
        val_kwargs = {
            **self.dataloader_kwargs,
            'shuffle': False,
            'pin_memory': self.dataloader_kwargs.get('pin_memory', False),
            'drop_last': False,
            'persistent_workers': True if self.num_workers > 0 else False,
        }
        return DataLoader(self.val_dataset, **val_kwargs)
    
    def test_dataloader(self):
        """Optimized test dataloader"""
        test_kwargs = {
            **self.dataloader_kwargs,
            'shuffle': False,
            'pin_memory': self.dataloader_kwargs.get('pin_memory', False),
            'drop_last': False,
            'persistent_workers': True if self.num_workers > 0 else False,
        }
        return DataLoader(self.test_dataset, **test_kwargs)

    def cleanup(self):
        """Clean up data and free memory"""
        if not self.is_cleaned_up:
            logger.info("Cleaning up DataModule resources...")
            self.dataset = None
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
                
            self.is_cleaned_up = True
            logger.info("DataModule cleanup completed")

    def teardown(self, stage: Optional[str] = None):
        """Called at the end of training"""
        logger.info("Teardown called")
        self.cleanup()


    def __del__(self):
        pass  
