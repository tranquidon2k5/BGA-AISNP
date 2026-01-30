"""Standardize AISNP_per_sample_with_pop.tsv to genotype counts
and **merge horizontally** with merged_matrix_201plus58_standardized.csv.

Inputs (expected in the same folder as this script):
- AISNP_per_sample_with_pop.tsv        (allele format _1/_2)
- merged_matrix_201plus58_standardized.csv (0/1/2 genotype counts)

Output:
- merged_matrix_201plus58_with_new_samples.csv

Behaviour:
- First, convert AISNP_per_sample_with_pop.tsv from allele format to
    0/1/2 genotype counts (same logic as standardize_new_data.py).
- Then merge **by sample ID** with merged_matrix_201plus58_standardized.csv
    so that:
    - Số dòng (số người) trong output = đúng số người của
        merged_matrix_201plus58_standardized.csv.
    - Chỉ thêm các SNP **mới** từ AISNP (các SNP đã có trong
        merged_matrix_201plus58_standardized.csv sẽ không được nhân đôi cột).
    - Các sample không có dữ liệu AISNP sẽ có giá trị NaN ở các cột SNP mới.
"""

import os
import numpy as np
import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

AISNP_FILE = os.path.join(THIS_DIR, "AISNP_per_sample_with_pop.tsv")
REF_FILE = os.path.join(THIS_DIR, "merged_matrix_201plus58_standardized.csv")
OUTPUT_FILE = os.path.join(THIS_DIR, "merged_matrix_201plus58_with_new_samples.csv")


def load_and_standardize_aisnp(path: str) -> pd.DataFrame:
    """Convert allele-format AISNP data to 0/1/2 genotype counts.

    The logic matches data/standardize_new_data.py but is factored
    here so we can then align columns to the reference matrix.
    """
    print(f"Loading AISNP allele data from: {path}")
    df = pd.read_csv(path, sep="\t")
    print(f"  AISNP raw shape: {df.shape}")

    metadata_cols = ["sample", "pop", "super_pop"]
    genotype_cols = [c for c in df.columns if c not in metadata_cols]

    df_meta = df[metadata_cols].copy()
    df_geno = df[genotype_cols].copy()

    # Collect unique SNP IDs (strip _1/_2 suffix)
    snp_names = sorted({c.rsplit("_", 1)[0] for c in genotype_cols})
    print(f"  Unique SNPs in AISNP data: {len(snp_names)}")

    geno_count_data = []
    valid_snps = []

    for snp in snp_names:
        col_1 = f"{snp}_1"
        col_2 = f"{snp}_2"
        if col_1 not in df_geno.columns or col_2 not in df_geno.columns:
            # skip incomplete SNPs
            continue

        allele_1 = df_geno[col_1].to_numpy()
        allele_2 = df_geno[col_2].to_numpy()

        all_alleles = np.concatenate([allele_1, allele_2])
        unique, counts = np.unique(all_alleles, return_counts=True)
        if len(unique) == 0:
            continue
        ref_allele = unique[np.argmax(counts)]

        # 0 = REF/REF, 1 = REF/ALT, 2 = ALT/ALT
        geno_count = np.zeros(len(allele_1), dtype=int)
        geno_count += (allele_1 != ref_allele).astype(int)
        geno_count += (allele_2 != ref_allele).astype(int)

        geno_count_data.append(geno_count)
        valid_snps.append(snp)

    df_geno_count = pd.DataFrame(
        np.column_stack(geno_count_data),
        columns=valid_snps,
    )

    df_standardized = pd.concat([df_meta.reset_index(drop=True), df_geno_count], axis=1)
    print(f"  AISNP standardized shape: {df_standardized.shape}")
    return df_standardized


def main() -> None:
    if not os.path.exists(AISNP_FILE):
        raise FileNotFoundError(f"AISNP file not found: {AISNP_FILE}")
    if not os.path.exists(REF_FILE):
        raise FileNotFoundError(f"Reference matrix not found: {REF_FILE}")

    # 1) Load and standardize AISNP allele data
    df_new = load_and_standardize_aisnp(AISNP_FILE)

    # 2) Load reference merged_matrix_201plus58_standardized
    print(f"\nLoading reference matrix from: {REF_FILE}")
    df_ref = pd.read_csv(REF_FILE)
    # Cột đầu tiên là sample ID (header trống), đổi tên thành "sample"
    first_col = df_ref.columns[0]
    if first_col != "sample":
        df_ref = df_ref.rename(columns={first_col: "sample"})

    print(f"  Reference shape (including pop/super_pop): {df_ref.shape}")

    metadata_cols = ["sample", "pop", "super_pop"]
    ref_snp_cols = [c for c in df_ref.columns if c not in metadata_cols]
    new_snp_cols = [c for c in df_new.columns if c not in metadata_cols]

    print(f"\nSNP columns in reference: {len(ref_snp_cols)}")
    print(f"SNP columns in new data: {len(new_snp_cols)}")

    # SNP trùng tên giữa ref và AISNP
    common_snps = sorted(set(ref_snp_cols).intersection(new_snp_cols))
    # SNP chỉ có trong AISNP -> sẽ được thêm mới theo chiều ngang
    extra_snps = sorted(set(new_snp_cols) - set(ref_snp_cols))

    print(f"  Common SNPs (chỉ giữ bản trong reference): {len(common_snps)}")
    if common_snps:
        print("    (ví dụ)", common_snps[:10])
    print(f"  SNPs chỉ có trong AISNP (sẽ thêm ngang): {len(extra_snps)}")
    if extra_snps:
        print("    (ví dụ)", extra_snps[:10])

    # 3) Chuẩn bị dataframe ref (giữ nguyên cột hiện tại)
    df_ref_aligned = df_ref[metadata_cols + ref_snp_cols].copy()

    # 4) Chuẩn bị dataframe AISNP chỉ với SNP mới + sample
    df_new_extra = df_new[["sample"] + extra_snps].copy()

    # Thống kê giao sample giữa 2 bảng
    ref_samples = set(df_ref_aligned["sample"])
    new_samples = set(df_new_extra["sample"])
    common_samples = ref_samples.intersection(new_samples)
    print(f"\nSamples in reference: {len(ref_samples)}")
    print(f"Samples in AISNP standardized: {len(new_samples)}")
    print(f"Common samples (sẽ được gộp ngang): {len(common_samples)}")

    # 5) Gộp NGANG: left-join theo sample, giữ nguyên số dòng của reference
    df_merged = df_ref_aligned.merge(df_new_extra, on="sample", how="left")
    print(f"\nMerged matrix shape (rows x cols): {df_merged.shape}")

    # 6) Save
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Merged matrix (horizontal) saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
