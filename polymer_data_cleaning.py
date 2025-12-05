import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def validate_smiles(smiles):
    """Validate SMILES string and return molecule object and validation info."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, 'invalid_rdkit_parsing'
        
        # Check for empty molecule
        if mol.GetNumAtoms() == 0:
            return None, 'empty_molecule'
            
        # Check for disconnected components (salts, mixtures)
        if '.' in smiles:
            return mol, 'disconnected_components'
            
        # All checks passed
        return mol, 'valid'
        
    except Exception as e:
        return None, f'parsing_error: {str(e)}'

def classify_polymer_molecule(smiles, mol):
    """Classify polymer molecule and identify potential issues."""
    issues = []
    
    # Check for wildcards (common in polymer SMILES)
    if '*' in smiles:
        wildcard_count = smiles.count('*')
        issues.append(f'wildcards_{wildcard_count}')
    
    # Check molecular weight (no upper limit for polymers, but flag very small ones)
    mol_wt = Descriptors.ExactMolWt(mol)
    if mol_wt < 50:  # Very small for a polymer
        issues.append('very_low_molecular_weight')
    
    # Check for very long SMILES (potential data quality issue)
    if len(smiles) > 1000:
        issues.append('very_long_smiles')
    
    # Check for unusual atom types for polymers
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    unusual_atoms = set(atoms) - {'C', 'N', 'O', 'H', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si'}
    if unusual_atoms:
        issues.append(f'unusual_atoms_{"|".join(unusual_atoms)}')
    
    # If no issues, mark as valid
    if not issues:
        return 'valid'
    else:
        return ';'.join(issues)

def analyze_property_patterns(df):
    """Analyze patterns in missing property data."""
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Create missing pattern analysis
    patterns = {}
    for _, row in df[properties].iterrows():
        pattern = tuple(pd.isna(row[prop]) for prop in properties)
        patterns[pattern] = patterns.get(pattern, 0) + 1
    
    return patterns

def create_stratified_split(df, test_size=0.15, val_size=0.15, random_state=42):
    """Create stratified splits based on data availability patterns."""
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Create a stratification key based on available properties
    df['available_props'] = df[properties].notna().sum(axis=1)
    df['has_complete_data'] = (df['available_props'] == 5)
    
    # Create bins for number of available properties
    df['prop_bin'] = pd.cut(df['available_props'], bins=[-0.5, 0.5, 2.5, 4.5, 5.5], 
                           labels=['none', 'few', 'some', 'complete'])
    
    try:
        # Stratified split based on property availability
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['prop_bin'], 
            random_state=random_state
        )
        
        # Further split train_val into train and validation
        train, val = train_test_split(
            train_val, 
            test_size=val_size/(1-test_size), 
            stratify=train_val['prop_bin'], 
            random_state=random_state
        )
        
    except ValueError as e:
        print(f"Stratified split failed: {e}")
        print("Falling back to random split...")
        
        # Fallback to random split
        train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)
    
    return train, val, test

def handle_missing_values_strategy(df, strategy='keep_all'):
    """Apply different strategies for handling missing values."""
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    if strategy == 'complete_only':
        # Keep only rows with all properties
        return df.dropna(subset=properties)
    
    elif strategy == 'at_least_one':
        # Keep rows with at least one property
        return df[df[properties].notna().any(axis=1)]
    
    elif strategy == 'at_least_three':
        # Keep rows with at least 3 properties
        return df[df[properties].notna().sum(axis=1) >= 3]
    
    elif strategy == 'keep_all':
        # Keep all rows (handle missing values in model)
        return df
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def clean_polymer_dataset(input_file, output_prefix, missing_strategy='keep_all', 
                         min_mol_weight=None, max_mol_weight=None):
    """Main data cleaning function for polymer dataset."""
    
    print("=" * 60)
    print("POLYMER DATASET CLEANING")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(input_file)
    original_count = len(df)
    print(f"Original dataset: {original_count} molecules")
    
    # Track cleaning steps
    cleaning_log = []
    
    # Step 1: Validate SMILES
    print("\nStep 1: Validating SMILES...")
    validation_results = []
    valid_indices = []
    
    for idx, smiles in enumerate(df['SMILES']):
        mol, status = validate_smiles(smiles)
        
        if mol is not None and status in ['valid', 'disconnected_components']:
            # Further classify the molecule
            classification = classify_polymer_molecule(smiles, mol)
            validation_results.append({
                'index': idx,
                'smiles': smiles,
                'status': status,
                'classification': classification,
                'mol_weight': Descriptors.ExactMolWt(mol),
                'num_atoms': mol.GetNumAtoms(),
                'has_wildcards': '*' in smiles,
                'num_wildcards': smiles.count('*')
            })
            
            # Decide whether to keep the molecule
            if classification == 'valid' or classification.startswith('wildcards_'):
                valid_indices.append(idx)
        else:
            validation_results.append({
                'index': idx,
                'smiles': smiles,
                'status': status,
                'classification': 'invalid',
                'mol_weight': None,
                'num_atoms': None,
                'has_wildcards': '*' in smiles,
                'num_wildcards': smiles.count('*') if isinstance(smiles, str) else 0
            })
    
    # Filter to valid molecules
    df_valid = df.iloc[valid_indices].copy()
    invalid_count = original_count - len(df_valid)
    cleaning_log.append(f"Removed {invalid_count} invalid SMILES")
    print(f"Valid SMILES: {len(df_valid)} (removed {invalid_count})")
    
    # Step 2: Apply molecular weight filters (if specified)
    if min_mol_weight is not None or max_mol_weight is not None:
        print(f"\nStep 2: Applying molecular weight filters...")
        validation_df = pd.DataFrame(validation_results)
        valid_validation = validation_df[validation_df['index'].isin(valid_indices)]
        
        mw_mask = pd.Series(True, index=df_valid.index)
        if min_mol_weight is not None:
            mw_mask &= (valid_validation.set_index('index')['mol_weight'] >= min_mol_weight)
        if max_mol_weight is not None:
            mw_mask &= (valid_validation.set_index('index')['mol_weight'] <= max_mol_weight)
        
        before_mw = len(df_valid)
        df_valid = df_valid[mw_mask]
        removed_mw = before_mw - len(df_valid)
        cleaning_log.append(f"Removed {removed_mw} molecules due to molecular weight filters")
        print(f"After MW filter: {len(df_valid)} (removed {removed_mw})")
    
    # Step 3: Handle missing values
    print(f"\nStep 3: Handling missing values (strategy: {missing_strategy})...")
    before_missing = len(df_valid)
    df_clean = handle_missing_values_strategy(df_valid, missing_strategy)
    removed_missing = before_missing - len(df_clean)
    cleaning_log.append(f"Removed {removed_missing} molecules due to missing value strategy")
    print(f"After missing value handling: {len(df_clean)} (removed {removed_missing})")
    
    # Step 4: Analyze final dataset
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    print(f"\nStep 4: Final dataset analysis...")
    
    for prop in properties:
        if prop in df_clean.columns:
            available = df_clean[prop].notna().sum()
            percentage = (available / len(df_clean)) * 100
            print(f"  {prop}: {available}/{len(df_clean)} ({percentage:.1f}%)")
    
    complete_cases = df_clean[properties].dropna().shape[0]
    print(f"  Complete cases: {complete_cases}/{len(df_clean)} ({complete_cases/len(df_clean)*100:.1f}%)")
    
    # Step 5: Create train/validation/test splits
    print(f"\nStep 5: Creating data splits...")
    train_df, val_df, test_df = create_stratified_split(df_clean)
    
    # Add group labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['group'] = 'training'
    val_df['group'] = 'valid'
    test_df['group'] = 'test'
    
    # Combine back
    final_df = pd.concat([train_df, val_df, test_df])
    
    print(f"Training set: {len(train_df)} molecules ({len(train_df)/len(final_df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} molecules ({len(val_df)/len(final_df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} molecules ({len(test_df)/len(final_df)*100:.1f}%)")
    
    # Step 6: Save results
    print(f"\nStep 6: Saving cleaned dataset...")
    
    # Save main cleaned dataset
    final_df.to_csv(f'{output_prefix}_cleaned.csv', index=False)
    
    # Save validation results
    validation_df = pd.DataFrame(validation_results)
    validation_df.to_csv(f'{output_prefix}_validation_log.csv', index=False)
    
    # Save filtering statistics
    excluded_df = df[~df.index.isin(final_df.index)]
    if len(excluded_df) > 0:
        excluded_df.to_csv(f'{output_prefix}_excluded.csv', index=False)
    
    # Create summary log
    with open(f'{output_prefix}_cleaning_log.txt', 'w') as f:
        f.write("POLYMER DATASET CLEANING LOG\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Original molecules: {original_count}\n")
        f.write(f"Final molecules: {len(final_df)}\n")
        f.write(f"Retention rate: {len(final_df)/original_count*100:.1f}%\n\n")
        
        f.write("Cleaning steps:\n")
        for step in cleaning_log:
            f.write(f"  - {step}\n")
        
        f.write(f"\nFinal split:\n")
        f.write(f"  Training: {len(train_df)} ({len(train_df)/len(final_df)*100:.1f}%)\n")
        f.write(f"  Validation: {len(val_df)} ({len(val_df)/len(final_df)*100:.1f}%)\n")
        f.write(f"  Test: {len(test_df)} ({len(test_df)/len(final_df)*100:.1f}%)\n")
        
        f.write(f"\nProperty availability in final dataset:\n")
        for prop in properties:
            if prop in final_df.columns:
                available = final_df[prop].notna().sum()
                percentage = (available / len(final_df)) * 100
                f.write(f"  {prop}: {available}/{len(final_df)} ({percentage:.1f}%)\n")
        
        complete_cases = final_df[properties].dropna().shape[0]
        f.write(f"\nComplete cases: {complete_cases}/{len(final_df)} ({complete_cases/len(final_df)*100:.1f}%)\n")
    
    print(f"\nCleaning complete! Files generated:")
    print(f"  - {output_prefix}_cleaned.csv (main dataset)")
    print(f"  - {output_prefix}_validation_log.csv (SMILES validation details)")
    print(f"  - {output_prefix}_cleaning_log.txt (summary log)")
    if len(excluded_df) > 0:
        print(f"  - {output_prefix}_excluded.csv (excluded molecules)")
    
    return final_df

def main():
    """Main function optimized for severe missing data scenario."""
    
    input_file = 'train.csv'
    
    print("POLYMER COMPETITION DATA CLEANING")
    print("Optimized for severe missing data scenario (0% complete cases)")
    print("=" * 60)
    
    # Use 'keep_all' strategy - essential for this competition
    strategy = 'keep_all'
    
    print(f"Using strategy: {strategy}")
    print("Rationale: With 0% complete cases, we need every available data point")
    
    # Clean dataset with no molecular weight limits (polymers can be very large)
    cleaned_df = clean_polymer_dataset(
        input_file=input_file,
        output_prefix='polymer_dataset',
        missing_strategy=strategy,
        min_mol_weight=None,  # No limits for polymers
        max_mol_weight=None
    )
    
    # Create special analysis for competition
    print("\n" + "="*60)
    print("COMPETITION-SPECIFIC ANALYSIS")
    print("="*60)
    
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Property availability analysis
    print("PROPERTY AVAILABILITY (Critical for strategy):")
    availability_dict = {}
    for prop in properties:
        available = cleaned_df[prop].notna().sum()
        total = len(cleaned_df)
        percentage = (available / total) * 100
        availability_dict[prop] = available / total
        print(f"  {prop}: {available:4d}/{total} ({percentage:5.1f}%) - {'PRIMARY' if percentage > 50 else 'SPARSE'}")
    
    # Missing pattern analysis (critical for model design)
    print(f"\nMISSING PATTERN ANALYSIS:")
    pattern_counts = {}
    for _, row in cleaned_df[properties].iterrows():
        available_props = [prop for prop in properties if not pd.isna(row[prop])]
        pattern = tuple(sorted(available_props))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print("Top 10 most common patterns:")
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (pattern, count) in enumerate(sorted_patterns[:10]):
        percentage = (count / len(cleaned_df)) * 100
        pattern_str = ', '.join(pattern) if pattern else 'No properties'
        print(f"  {i+1:2d}. {pattern_str:30s}: {count:4d} ({percentage:5.1f}%)")
    
    # Strategy recommendations based on actual data
    print(f"\nMODEL STRATEGY RECOMMENDATIONS:")
    
    # Find primary property (most available)
    primary_prop = max(availability_dict.items(), key=lambda x: x[1])
    print(f"Primary property: {primary_prop[0]} ({primary_prop[1]*100:.1f}% coverage)")
    
    # Find sparse properties
    sparse_props = [prop for prop, avail in availability_dict.items() if avail < 0.2]
    print(f"Sparse properties: {sparse_props}")
    
    # Calculate property weights for loss function
    print(f"\nRECOMMENDED LOSS WEIGHTS:")
    min_availability = min(availability_dict.values())
    for prop in properties:
        weight = min_availability / availability_dict[prop]
        print(f"  {prop}: {weight:.3f}")
    
    # Training set analysis
    print(f"\nTRAINING SET DISTRIBUTION:")
    for group in ['training', 'valid', 'test']:
        group_data = cleaned_df[cleaned_df['group'] == group]
        print(f"\n{group.upper()} SET ({len(group_data)} molecules):")
        
        for prop in properties:
            available = group_data[prop].notna().sum()
            percentage = (available / len(group_data)) * 100
            print(f"    {prop}: {available:3d}/{len(group_data)} ({percentage:5.1f}%)")
    
    # Competition submission preparation
    print(f"\nCOMPETITION SUBMISSION NOTES:")
    print("- All test molecules will likely have wildcard atoms (*)")
    print("- Test set will have unknown property availability pattern")
    print("- Model must predict all 5 properties regardless of training availability")
    print("- Competition metric is weighted MAE - weights unknown until final scoring")
    
    # Critical warnings
    print(f"\nCRITICAL CONSIDERATIONS:")
    print("⚠️  0% complete cases - Multi-task learning with heavy missing value handling required")
    print("⚠️  Severe class imbalance in property availability")
    print("⚠️  All molecules have wildcards - ensure proper graph construction")
    print("⚠️  Wide molecular weight range (14-2200) - may need property-specific normalization")
    

    wildcards = cleaned_df['SMILES'].str.count(r'\*')
    percent_wildcards = (wildcards > 0).sum() / len(cleaned_df) * 100
    print(f"Wildcards present in {percent_wildcards:.1f}% of molecules")
    print(f"Max wildcards per molecule: {wildcards.max()}")

    return cleaned_df

    
    # Recommendations for model development
    print(f"\nRECOMMENDATIONS FOR RGCN MODEL:")
    complete_cases = cleaned_df[properties].dropna().shape[0]
    total_cases = len(cleaned_df)
    
    if complete_cases / total_cases < 0.1:
        print("- Implement multi-task learning with missing value masking")
        print("- Consider separate models for each property")
        print("- Use property-specific loss weighting")
    elif complete_cases / total_cases < 0.5:
        print("- Multi-task learning with missing value handling recommended")
        print("- Implement property masking in loss calculation")
        print("- Consider ensemble of property-specific models")
    else:
        print("- Multi-task learning fully feasible")
        print("- Can use joint optimization for all properties")
    
    if (wildcards > 0).sum() > 0:
        print("- Ensure proper handling of wildcard atoms (*) in graph construction")
        print("- Consider masking wildcard atoms or special encoding")
    
    print("- Implement weighted loss based on property availability")
    print("- Use stratified sampling to maintain property distribution")
    
    return cleaned_df

if __name__ == "__main__":
    main()