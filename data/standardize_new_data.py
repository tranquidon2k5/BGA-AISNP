"""
Standardize new AISNP data from allele format to genotype count format
Input: AISNP_per_sample_with_pop.tsv (allele format: _1, _2 for each SNP)
Output: AISNP_standardized.csv (genotype count format: 0/1/2)
"""

import pandas as pd
import numpy as np
import os

# Load data
data_path = 'AISNP_per_sample_with_pop.tsv'
df = pd.read_csv(data_path, sep='\t')

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:10]}... (showing first 10)")

# Separate metadata and genotypes
metadata_cols = ['sample', 'pop', 'super_pop']
genotype_cols = [col for col in df.columns if col not in metadata_cols]

print(f"\nMetadata columns: {metadata_cols}")
print(f"Genotype columns: {len(genotype_cols)} (allele format)")

# Extract SNP names and metadata
df_meta = df[metadata_cols].copy()
df_geno = df[genotype_cols].copy()

# Parse SNP names from allele columns
snp_names_set = set()
for col in genotype_cols:
    snp = col.rsplit('_', 1)[0]  # Remove _1 or _2 suffix
    snp_names_set.add(snp)

snp_names_list = sorted(snp_names_set)
print(f"\nUnique SNPs: {len(snp_names_list)}")
print(f"First 10 SNPs: {snp_names_list[:10]}")

# Convert from allele format to genotype count
print(f"\nConverting from allele format to genotype count format...")

geno_count_data = []
for snp in snp_names_list:
    col_1 = f"{snp}_1"
    col_2 = f"{snp}_2"
    
    if col_1 in df_geno.columns and col_2 in df_geno.columns:
        # Get allele values
        allele_1 = df_geno[col_1].values
        allele_2 = df_geno[col_2].values
        
        # Determine REF allele (most common allele across the dataset)
        all_alleles = np.concatenate([allele_1, allele_2])
        unique, counts = np.unique(all_alleles, return_counts=True)
        ref_allele = unique[np.argmax(counts)]
        
        # Count ALT alleles (number of non-REF alleles)
        # 0 = REF/REF, 1 = REF/ALT, 2 = ALT/ALT
        geno_count = np.zeros(len(allele_1), dtype=int)
        geno_count += (allele_1 != ref_allele).astype(int)
        geno_count += (allele_2 != ref_allele).astype(int)
        
        geno_count_data.append(geno_count)

# Create DataFrame with genotype counts
df_geno_count = pd.DataFrame(
    np.column_stack(geno_count_data),
    columns=snp_names_list
)

# Combine metadata and genotypes
df_standardized = pd.concat([df_meta.reset_index(drop=True), df_geno_count], axis=1)

print(f"\nStandardized data shape: {df_standardized.shape}")
print(f"Unique genotype values: {sorted(np.unique(df_standardized[snp_names_list].values))}")

# Check for missing values
missing_count = df_standardized[snp_names_list].isnull().sum().sum()
print(f"Missing values: {missing_count}")

# Save
output_path = 'AISNP_standardized.csv'
df_standardized.to_csv(output_path, index=False)
print(f"\n✓ Standardized data saved to {output_path}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Samples: {len(df_standardized):,}")
print(f"SNPs: {len(snp_names_list)}")
print(f"Populations: {df_standardized['pop'].nunique()}")
print(f"Continents: {df_standardized['super_pop'].nunique()}")
print(f"\nPopulation distribution:")
print(df_standardized['pop'].value_counts().sort_index())
print(f"\nContinent distribution:")
print(df_standardized['super_pop'].value_counts().sort_index())
print(f"{'='*70}")
