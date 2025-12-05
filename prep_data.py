import argparse
import os
import pandas as pd
import traceback
from typing import List
from build_data import (
    build_mol_graph_data,
    build_mol_graph_data_for_brics,
    build_mol_graph_data_for_murcko,
    build_mol_graph_data_for_fg,
    save_dataset
)
from logger import get_logger
from config import Configuration

# Get configuration
config = Configuration()

# Get logger
logger = get_logger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare molecular graph datasets.")
    
    parser.add_argument(
        '--task_name',
        type=str,
        default=config.task_name,
        help=f"Name of the task (default: {config.task_name})"
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=config.origin_data_path,
        help=f"Path to the input CSV dataset file (default: {config.origin_data_path})"
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=config.output_dir,
        help=f"Directory where processed data will be saved (default: {config.output_dir})"
    )
    
    parser.add_argument(
        '--force_rebuild',
        action='store_true',
        help="Force rebuild of all graph data even if files exist"
    )
    
    parser.add_argument(
        '--substructure_types',
        nargs='+',
        default=config.substructure_types,
        choices=['primary', 'brics', 'murcko', 'fg'],
        help=f"List of substructure types to process (default: {config.substructure_types})"
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.batch_size,
        help=f"Batch size for processing (default: {config.batch_size})"
    )
    
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args

def check_files_exist(config: Configuration, substructure_types: List[str]) -> bool:
    """Check if all required files already exist."""
    logger.info("Checking for existing files...")
    
    # Determine task type based on configuration
    task_type = 'classification' if config.classification else 'regression'
    
    all_exist = True
    for subtype in substructure_types:
        graph_path = os.path.join(config.output_dir, f"{config.task_name}_{task_type}_{subtype}_graphs.pt")
        meta_path = os.path.join(config.output_dir, f"{config.task_name}_{task_type}_{subtype}_meta.csv")
        
        if not os.path.exists(graph_path):
            logger.info(f"Missing graph file for {subtype}: {graph_path}")
            all_exist = False
        if not os.path.exists(meta_path):
            logger.info(f"Missing meta file for {subtype}: {meta_path}")
            all_exist = False
            
    return all_exist

def prepare_data(config: Configuration, force_rebuild: bool = False, substructure_types: List[str] = None) -> bool:
    """Prepare molecular graph datasets."""
    try:
        # Log the start of dataset inspection
        logger.info(f"Starting data preparation with:")
        logger.info(f"- Task name: {config.task_name}")
        logger.info(f"- Input file: {config.origin_data_path}")
        logger.info(f"- Output directory: {config.output_dir}")
        logger.info(f"- Substructure types: {substructure_types}")
        
        # First inspect the dataset to determine task type
        logger.info("Inspecting dataset...")
        config.inspect_dataset()
        logger.info(f"Dataset inspection complete. Task type: {config.task_type}")
        
        # Check if files exist
        if not force_rebuild:
            files_exist = check_files_exist(config, substructure_types)
            if files_exist:
                logger.info("All data files already exist. Use --force_rebuild to regenerate.")
                return True
            logger.info("Some files are missing, proceeding with data preparation.")
        
        # Dictionary mapping substructure types to their processing functions
        processors = {
            'primary': (build_mol_graph_data, False),
            'brics': (build_mol_graph_data_for_brics, False),
            'murcko': (build_mol_graph_data_for_murcko, False),
            'fg': (build_mol_graph_data_for_fg, True)
        }
        
        # Process each substructure type
        for subtype in substructure_types:
            logger.info(f"\nProcessing {subtype} substructure type...")
            
            if subtype not in processors:
                logger.warning(f"Skipping unknown substructure type: {subtype}")
                continue
            
            processor_func, needs_config = processors[subtype]
            
            try:
                logger.info(f"Building {subtype} graph data...")
                if needs_config:
                    dataset = processor_func(
                        config.dataset_origin,
                        config.labels_name,
                        config.smiles_name,
                        config
                    )
                else:
                    dataset = processor_func(
                        config.dataset_origin,
                        config.labels_name,
                        config.smiles_name
                    )
                
                # Get paths using the configuration helper method
                graph_path = config.get_processed_file_path('graphs', subtype)
                meta_path = config.get_processed_file_path('meta', subtype)
                smask_path = config.get_processed_file_path('smask', subtype) if subtype != 'primary' else None
                
                logger.info(f"Saving {subtype} data...")
                logger.info(f"- Graph path: {graph_path}")
                logger.info(f"- Meta path: {meta_path}")
                if smask_path:
                    logger.info(f"- Smask path: {smask_path}")
                
                # Save the dataset
                save_dataset(
                    dataset=dataset,
                    sub_type=subtype,
                    config=config,
                    graph_path=graph_path,
                    meta_path=meta_path,
                    smask_save_path=smask_path
                )
                
                logger.info(f"Successfully processed {subtype} substructure type")
                
            except Exception as e:
                logger.error(f"Error processing {subtype} substructure type:")
                logger.error(traceback.format_exc())
                raise
        
        logger.info("\nData preparation completed successfully")
        return True
        
    except Exception as e:
        logger.error("Error during data preparation:")
        logger.error(traceback.format_exc())
        raise

def main():
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Log startup information
        logger.info("Starting data preparation script")
        logger.info(f"Arguments: {args}")
        
        # Update configuration with command line arguments
        if args.task_name:
            config.task_name = args.task_name
        if args.dataset_path:
            config.origin_data_path = args.dataset_path
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.batch_size:
            config.batch_size = args.batch_size
        
        # Validate configuration
        logger.info("Validating configuration...")
        config.validate()
        logger.info("Configuration validation successful")
        
        # Set random seed for reproducibility
        config.set_seed(seed=42)
        logger.info("Random seed set to 42")
        
        # Prepare data
        logger.info(f"Starting data preparation for task: {config.task_name}")
        success = prepare_data(config, args.force_rebuild, args.substructure_types)
        
        if success:
            logger.info("Data preparation completed successfully")
            # Verify created files
            logger.info("Verifying created files:")
            for subtype in args.substructure_types:
                graph_path = config.get_processed_file_path('graphs', subtype)
                meta_path = config.get_processed_file_path('meta', subtype)
                logger.info(f"{subtype} files:")
                logger.info(f"- Graph file exists: {os.path.exists(graph_path)}")
                logger.info(f"- Meta file exists: {os.path.exists(meta_path)}")
        else:
            logger.error("Data preparation failed")
            
    except Exception as e:
        logger.error("Fatal error during execution:")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()