import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')

def load_and_inspect_data(file_path):
    """Load data and perform basic inspection."""
    print("=" * 60)
    print("POLYMER DATASET EDA")
    print("=" * 60)
    
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def analyze_missing_values(df):
    """Analyze missing values pattern for polymer properties."""
    print("\n" + "="*50)
    print("MISSING VALUES ANALYSIS")
    print("="*50)
    
    # Define the 5 polymer properties
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    missing_info = []
    for prop in properties:
        if prop in df.columns:
            missing_count = df[prop].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            available_count = len(df) - missing_count
            missing_info.append({
                'Property': prop,
                'Available': available_count,
                'Missing': missing_count,
                'Missing_%': missing_pct
            })
        else:
            print(f"Warning: Property {prop} not found in dataset")
    
    missing_df = pd.DataFrame(missing_info)
    print("\nMissing Values Summary:")
    print(missing_df)
    
    # Analyze patterns of missing values
    print("\nMissing Value Patterns:")
    df_props = df[properties].copy()
    
    # Count complete cases
    complete_cases = df_props.dropna().shape[0]
    print(f"Complete cases (all 5 properties): {complete_cases} ({complete_cases/len(df)*100:.1f}%)")
    
    # Pattern analysis
    patterns = {}
    for _, row in df_props.iterrows():
        pattern = tuple(pd.isna(row[prop]) for prop in properties)
        patterns[pattern] = patterns.get(pattern, 0) + 1
    
    print("\nTop missing patterns:")
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    for i, (pattern, count) in enumerate(sorted_patterns[:10]):
        missing_props = [properties[j] for j, is_missing in enumerate(pattern) if is_missing]
        available_props = [properties[j] for j, is_missing in enumerate(pattern) if not is_missing]
        print(f"{i+1:2d}. Count: {count:4d} | Missing: {missing_props} | Available: {available_props}")
    
    return missing_df, complete_cases

def analyze_smiles(df):
    """Analyze SMILES patterns and validity."""
    print("\n" + "="*50)
    print("SMILES ANALYSIS")
    print("="*50)
    
    smiles_col = 'SMILES'
    valid_smiles = []
    invalid_smiles = []
    polymer_characteristics = []
    
    for idx, smiles in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            invalid_smiles.append((idx, smiles, "Invalid SMILES"))
            continue
            
        valid_smiles.append(idx)
        
        # Analyze polymer characteristics
        characteristics = {
            'index': idx,
            'smiles': smiles,
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'mol_weight': Descriptors.ExactMolWt(mol),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'has_wildcard': '*' in smiles,
            'num_wildcards': smiles.count('*'),
            'smiles_length': len(smiles),
            'has_stereochemistry': '@' in smiles,
            'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol)
        }
        polymer_characteristics.append(characteristics)
    
    print(f"Valid SMILES: {len(valid_smiles)} ({len(valid_smiles)/len(df)*100:.1f}%)")
    print(f"Invalid SMILES: {len(invalid_smiles)} ({len(invalid_smiles)/len(df)*100:.1f}%)")
    
    if invalid_smiles:
        print("\nInvalid SMILES examples:")
        for idx, smiles, reason in invalid_smiles[:5]:
            print(f"  Row {idx}: {smiles[:50]}... - {reason}")
    
    # Create DataFrame for analysis
    char_df = pd.DataFrame(polymer_characteristics)
    
    if len(char_df) > 0:
        print(f"\nPolymer Structure Statistics:")
        print(f"Molecular weight range: {char_df['mol_weight'].min():.1f} - {char_df['mol_weight'].max():.1f}")
        print(f"Number of atoms range: {char_df['num_atoms'].min()} - {char_df['num_atoms'].max()}")
        print(f"SMILES length range: {char_df['smiles_length'].min()} - {char_df['smiles_length'].max()}")
        print(f"Polymers with wildcards (*): {char_df['has_wildcard'].sum()} ({char_df['has_wildcard'].sum()/len(char_df)*100:.1f}%)")
        print(f"Polymers with stereochemistry: {char_df['has_stereochemistry'].sum()} ({char_df['has_stereochemistry'].sum()/len(char_df)*100:.1f}%)")
    
    return char_df, valid_smiles, invalid_smiles

def analyze_property_distributions(df):
    """Analyze the distributions of polymer properties."""
    print("\n" + "="*50)
    print("PROPERTY DISTRIBUTIONS")
    print("="*50)
    
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    available_props = [prop for prop in properties if prop in df.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, prop in enumerate(available_props):
        if i < len(axes):
            data = df[prop].dropna()
            if len(data) > 0:
                axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{prop} Distribution\n(n={len(data)})')
                axes[i].set_xlabel(prop)
                axes[i].set_ylabel('Frequency')
                
                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                std_val = data.std()
                axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
                axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.3f}')
                axes[i].legend(fontsize=8)
                
                print(f"\n{prop} Statistics:")
                print(f"  Count: {len(data)}")
                print(f"  Mean: {mean_val:.4f}")
                print(f"  Std: {std_val:.4f}")
                print(f"  Min: {data.min():.4f}")
                print(f"  Max: {data.max():.4f}")
            else:
                axes[i].text(0.5, 0.5, f'No data for {prop}', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{prop} - No Data')
    
    # Remove empty subplots
    for i in range(len(available_props), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('property_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_property_correlations(df):
    """Analyze correlations between polymer properties."""
    print("\n" + "="*50)
    print("PROPERTY CORRELATIONS")
    print("="*50)
    
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    available_props = [prop for prop in properties if prop in df.columns]
    
    # Calculate correlations only for samples with complete data
    complete_data = df[available_props].dropna()
    
    if len(complete_data) > 1:
        correlation_matrix = complete_data.corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(f'Property Correlations\n(Based on {len(complete_data)} complete samples)')
        plt.tight_layout()
        plt.savefig('property_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Correlation analysis based on {len(complete_data)} complete samples")
        print("\nStrong correlations (|r| > 0.5):")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    prop1 = correlation_matrix.columns[i]
                    prop2 = correlation_matrix.columns[j]
                    print(f"  {prop1} - {prop2}: {corr_val:.3f}")
    else:
        print("Insufficient complete data for correlation analysis")

def analyze_data_completeness_by_structure(df, char_df):
    """Analyze data completeness patterns by molecular structure characteristics."""
    print("\n" + "="*50)
    print("DATA COMPLETENESS BY STRUCTURE")
    print("="*50)
    
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Merge structure characteristics with property data
    if len(char_df) > 0:
        merged_df = df.merge(char_df[['index', 'mol_weight', 'num_atoms', 'has_wildcard', 'num_wildcards']], 
                           left_index=True, right_on='index', how='left')
        
        # Count available properties per molecule
        merged_df['num_available_props'] = merged_df[properties].notna().sum(axis=1)
        
        # Analyze by molecular weight bins
        if 'mol_weight' in merged_df.columns and merged_df['mol_weight'].notna().sum() > 0:
            merged_df['mw_bin'] = pd.cut(merged_df['mol_weight'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            mw_completeness = merged_df.groupby('mw_bin')['num_available_props'].agg(['count', 'mean', 'std'])
            print("\nData completeness by molecular weight:")
            print(mw_completeness)
        
        # Analyze by wildcard presence
        if 'has_wildcard' in merged_df.columns:
            wildcard_completeness = merged_df.groupby('has_wildcard')['num_available_props'].agg(['count', 'mean', 'std'])
            print("\nData completeness by wildcard presence:")
            # Only rename index if we have both True and False values
            if len(wildcard_completeness) == 2:
                wildcard_completeness.index = ['No Wildcards', 'Has Wildcards']
            elif len(wildcard_completeness) == 1 and wildcard_completeness.index[0] == True:
                wildcard_completeness.index = ['All Have Wildcards']
            elif len(wildcard_completeness) == 1 and wildcard_completeness.index[0] == False:
                wildcard_completeness.index = ['None Have Wildcards']
            print(wildcard_completeness)

def generate_summary_report(df, missing_df, complete_cases, char_df, valid_smiles, invalid_smiles):
    """Generate a comprehensive summary report."""
    print("\n" + "="*50)
    print("SUMMARY REPORT")
    print("="*50)
    
    total_molecules = len(df)
    valid_smiles_count = len(valid_smiles)
    
    print(f"Total molecules in dataset: {total_molecules}")
    print(f"Valid SMILES: {valid_smiles_count} ({valid_smiles_count/total_molecules*100:.1f}%)")
    print(f"Complete property data: {complete_cases} ({complete_cases/total_molecules*100:.1f}%)")
    
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    print(f"\nProperty availability:")
    for _, row in missing_df.iterrows():
        print(f"  {row['Property']}: {row['Available']} samples ({100-row['Missing_%']:.1f}%)")
    
    print(f"\nRecommendations for model development:")
    if complete_cases < total_molecules * 0.1:
        print("  - Very low complete data rate. Consider multi-task learning with missing value handling")
        print("  - Implement separate models for each property or use imputation strategies")
    elif complete_cases < total_molecules * 0.5:
        print("  - Moderate complete data rate. Multi-task learning feasible with proper handling")
        print("  - Consider ensemble approaches combining property-specific models")
    else:
        print("  - Good complete data rate. Multi-task learning recommended")
    
    if len(invalid_smiles) > 0:
        print(f"  - Clean {len(invalid_smiles)} invalid SMILES before model training")
    
    if len(char_df) > 0 and char_df['has_wildcard'].sum() > 0:
        print(f"  - {char_df['has_wildcard'].sum()} polymers contain wildcards (*) - ensure proper handling")

def main():
    """Main EDA function."""
    # Load data
    file_path = 'train.csv'  # Update with your file path
    df = load_and_inspect_data(file_path)
    
    # Analyze missing values
    missing_df, complete_cases = analyze_missing_values(df)
    
    # Analyze SMILES
    char_df, valid_smiles, invalid_smiles = analyze_smiles(df)
    
    # Analyze property distributions
    analyze_property_distributions(df)
    
    # Analyze correlations
    analyze_property_correlations(df)
    
    # Analyze completeness by structure
    analyze_data_completeness_by_structure(df, char_df)
    
    # Generate summary
    generate_summary_report(df, missing_df, complete_cases, char_df, valid_smiles, invalid_smiles)
    
    # Save analysis results
    if len(char_df) > 0:
        char_df.to_csv('molecular_characteristics.csv', index=False)
    missing_df.to_csv('missing_values_analysis.csv', index=False)
    
    print(f"\nAnalysis complete! Check generated files:")
    print("  - molecular_characteristics.csv")
    print("  - missing_values_analysis.csv")
    print("  - property_distributions.png")
    print("  - property_correlations.png")

if __name__ == "__main__":
    main()